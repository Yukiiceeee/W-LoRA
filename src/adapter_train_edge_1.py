from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Optional, Sequence, List
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from safetensors.torch import load_file
from dataclasses import dataclass, field
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
import json
import torch
import os
import copy
import logging
import transformers
from dataset import mcDataset
from dataset import haDataset
from dataset import ScienceQADataset

MyDataset = None
DATASET_PATH = None
DATASET_NAME = None
CANDIDATE_DATASETS = {
    "medmc":        [mcDataset,      "/d2/mxy/W-LoRA/data/Domains-Based/med/mc"],
    "finha":        [haDataset,      "/d2/mxy/W-LoRA/data/Domains-Based/fin/ha"],
    "scienceqa":    [ScienceQADataset, "/d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf"]
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/d2/mxy/Models/Qwen2-7B")
    peft_type: Optional[str] = field(default="lora")
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)

    incremental_lora: Optional[int] = field(default=1)
    base_lora_path: Optional[str] = field(default=None)
    base_lora_r: Optional[int] = field(default=None)
    freeze_base_lora: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    data_name: str = field(default="medmc", metadata={"help": "data name"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="./cache")
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=2e-4)
    lr_scheduler_type: str = field(default="cosine")
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    per_device_eval_batch_size: int = field(default=16)
    num_train_epochs: int = field(default=10)
    weight_decay: float = field(default=0.01)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=2)
    logging_dir: str = field(default="./logs")
    logging_steps: int = field(default=10)
    use_deepspeed: bool = field(default=False)
    output_dir: Optional[str] = field(default="./results", metadata={"help": "Path to the output directory"})

def get_target_modules(model_type, named_modules):
    target_modules = LORA_TARGET_MAP.get(model_type, [])
    if "qwen" in model_type.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # if not target_modules:
    #     cls = torch.nn.Linear
    #     lora_module_names = {name.split('.')[-1] for name, module in named_modules if isinstance(module, cls)}
    #     if "lm_head" in lora_module_names:

def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "torch_dtype": 'auto',
        "trust_remote_code": True,
    }
    if not training_args.use_deepspeed:
        model_kwargs["device_map"] = "auto"
    else:
        logger.warning("Using DeepSpeed")

    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )

    # 获取目标模块列表
    target_modules = get_target_modules(model.config.model_type.lower(), model.named_modules())

    # 创建新LoRA配置（总秩r=10）
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        inference_mode=False,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    
    logger.info("Original trainable parameters:")
    model.print_trainable_parameters()

    if model_args.incremental_lora and model_args.base_lora_path and model_args.base_lora_r:
        logger.info(f"Loading base LoRA weights from {model_args.base_lora_path} with r={model_args.base_lora_r}")
        base_model_path = os.path.join(model_args.base_lora_path, "adapter_model.safetensors")
        base_model_state_dict = load_file(base_model_path)

        # 构建名称映射表
        name_mapping = {}
        for old_name in base_model_state_dict.keys():
            # 移除可能存在的多余前缀
            new_name = old_name.replace("base_model.model.", "")
            new_name = new_name.replace(".default", "")
            name_mapping[new_name] = old_name

        # 参数处理
        for name, param in model.named_parameters():
            if 'lora_' in name:
                # 转换参数名
                simplified_name = name.replace("base_model.model.", "").replace(".default", "")
                base_weight_name = name_mapping.get(simplified_name, None)

                if not base_weight_name:
                    logger.warning(f"Missing mapping for {name}")
                    continue

                if base_weight_name and base_weight_name in base_model_state_dict:
                    base_weight = base_model_state_dict[base_weight_name]
                    
                    if base_weight is None:
                        logger.warning(f"Missing base weight for {base_weight_name}")
                        continue

                    # 处理A矩阵
                    if 'lora_A' in name:
                        # 维度校验
                        if base_weight.shape[0] != model_args.base_lora_r:
                            raise ValueError(f"A矩阵维度不匹配: {base_weight.shape} vs 预期r={model_args.base_lora_r}")
                        
                        # 加载权重并设置梯度
                        param.data[:model_args.base_lora_r] = base_weight
                        grad_mask = torch.ones_like(param, dtype=torch.bool)
                        grad_mask[:model_args.base_lora_r] = False
                        param.requires_grad = True
                        param.register_hook(lambda grad: grad * grad_mask.to(grad.device))

                    # 处理B矩阵
                    elif 'lora_B' in name:
                        if base_weight.shape[1] != model_args.base_lora_r:
                            raise ValueError(f"B矩阵维度不匹配: {base_weight.shape} vs 预期r={model_args.base_lora_r}")
                        
                        param.data[:, :model_args.base_lora_r] = base_weight
                        grad_mask = torch.ones_like(param, dtype=torch.bool)
                        grad_mask[:, :model_args.base_lora_r] = False
                        param.requires_grad = True
                        param.register_hook(lambda grad: grad * grad_mask.to(grad.device))

        # 正确统计参数
        logger.info("\nFinal trainable parameters:")
        model.print_trainable_parameters()

        model.train()
        dummy_input = tokenizer("Test input for grad check", return_tensors="pt").to(model.device)
        dummy_input["labels"] = dummy_input["input_ids"].clone()
        outputs = model(**dummy_input)
        loss = outputs.loss
        # 反向传播
        loss.backward()

        # 验证冻结效果
        total_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if 'lora_' in name:
                total_params += param.numel()
                if param.grad is None:  # 未计算过梯度的参数
                    frozen_params += param.numel()
                else:  # 计算过梯度但被mask的参数
                    frozen_params += (param.grad == 0).sum().item()
        logger.info(f"实际冻结参数比例: {100*frozen_params/total_params:.2f}%")

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    return model, tokenizer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    incremental_lora = model_args.incremental_lora

    if model_args.incremental_lora:
        if not model_args.base_lora_path:
            raise ValueError("base_lora_path must be provided when incremental_lora=True")
        if not model_args.base_lora_r:
            raise ValueError("base_lora_r must be provided when incremental_lora=True")
        if model_args.base_lora_r >= model_args.lora_r:
            raise ValueError(f"base_lora_r ({model_args.base_lora_r}) must be less than lora_r ({model_args.lora_r})")
        logger.info(f"Incremental LoRA training: loading r={model_args.base_lora_r} from {model_args.base_lora_path}, training to r={model_args.lora_r}")
    
    global DATASET_NAME, MyDataset, DATASET_PATH
    DATASET_NAME = data_args.data_name
    MyDataset = CANDIDATE_DATASETS[DATASET_NAME][0]
    DATASET_PATH = CANDIDATE_DATASETS[DATASET_NAME][1]

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    ### Data processing
    train_dataset, test_dataset = MyDataset.load_dataset(
        data_path = DATASET_PATH,
        tokenizer = tokenizer,
        max_length = training_args.model_max_length,
        incremental_lora = incremental_lora
    )
    logger.warning(f"train_dataset numbers: {len(train_dataset)}")
    logger.warning(f"test_dataset numbers: {len(test_dataset)}")

    ### Training
    logger.warning("Creating trainer...")
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset
    )
    logger.warning("Training...")
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    logger.info(f"Saved adapter successfully")
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def main():
    train()

if __name__ == "__main__":
    main()
    