from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftMixedModel
from typing import Dict, Optional, Sequence, List
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
import json
import torch
import os
import copy
import logging
import transformers
from safetensors import safe_open
from dataset import mcDataset
from dataset import haDataset
from dataset import ScienceQADataset
from typing import Dict, Optional, Sequence, List
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
import json
import torch
import os
import copy
import logging
import transformers
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftMixedModel
import re
from tqdm import tqdm
from constants import PROMPT_DICT
from datasets import load_from_disk
import numpy as np

def monkey_patch_peft_mixed_model():
    def get_base_model(self):
        return self.model
    
    PeftMixedModel.get_base_model = get_base_model

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

def log_parameter_stats(name, param):
    if param.numel() == 0:
        logger.info(f"{name}: Empty tensor")
        return
    
    data = param.detach().cpu().float().numpy()
    non_zero = np.count_nonzero(data)
    zero_ratio = 1.0 - (non_zero / data.size)
    
    logger.info(f"{name} stats: "
                f"mean={np.mean(data):.6f}, "
                f"std={np.std(data):.6f}, "
                f"min={np.min(data):.6f}, "
                f"max={np.max(data):.6f}, "
                f"zeros={zero_ratio*100:.2f}%")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/d2/mxy/Models/Qwen2-7B")
    peft_type: Optional[str] = field(default="lora")
    lora_r: Optional[int] = field(default=10)
    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    incremental_lora:Optional[str] = field(default="domain")
    base_lora_r: Optional[int] = field(default=8)
    base_lora_path: Optional[str] = field(default=None)

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

class ParameterTrackingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_params_before = {}
        self.incr_params_before = {}
        self.check_interval = 100
        
        logger.info("保存初始参数状态...")
        for name, param in self.model.named_parameters():
            if "base_adapter" in name:
                self.base_params_before[name] = param.data.clone().cpu()
            elif "incremental_adapter" in name:
                self.incr_params_before[name] = param.data.clone().cpu()
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if self.state.global_step % self.check_interval == 0:
            self.check_parameter_updates()
            
        return loss
    
    def check_parameter_updates(self):
        logger.info(f"步骤 {self.state.global_step}: 参数更新...")
        
        # for name, param in self.model.named_parameters():
        #     if "base_adapter" in name and name in self.base_params_before:
        #         original = self.base_params_before[name]
        #         current = param.data.cpu()
        #         param_diff = torch.abs(original - current).mean().item()
        #         logger.info(f"{name} 参数变化量: {param_diff:.8f}")
                
        #         if param_diff > 1e-8:
        #             logger.warning(f"警告: base_adapter参数 {name} 发生了变化!")
                
        #     elif "incremental_adapter" in name and name in self.incr_params_before:
        #         original = self.incr_params_before[name]
        #         current = param.data.cpu()
        #         param_diff = torch.abs(original - current).mean().item()
        #         logger.info(f"{name} 参数变化量: {param_diff:.8f}")

def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    model_kwargs={
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
        padding_side = "left",
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )
    target_modules = get_target_modules(model.config.model_type.lower(), model.named_modules())

    if model_args.incremental_lora == "task":
        old_config = LoraConfig(
            r = model_args.base_lora_r,
            lora_alpha = model_args.lora_alpha,
            lora_dropout = model_args.lora_dropout,
            inference_mode = False,
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules=target_modules,
        )
        # model = get_peft_model(model, old_config, adapter_name="base_adapter")
        # model = PeftModel(model, old_config, adapter_name="base_adapter")
        model = PeftMixedModel(model, old_config, adapter_name="base_adapter")
        
        # 检查1: 记录加载前base_adapter参数状态
        logger.info("***检查1: 记录加载base_adapter参数前的状态")
        base_params_before = {}
        for name, param in model.named_parameters():
            if "base_adapter" in name:
                if "lora_A" in name or "lora_B" in name:
                    log_parameter_stats(name, param)
                    base_params_before[name] = param.data.clone()
        
        logger.info(f"正在从 {model_args.base_lora_path} 加载base_adapter权重...")
        adapter_weights_path = f"{model_args.base_lora_path}/adapter_model.safetensors"
        adapter_state_dict = {}
        with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = key.replace(".lora_A.", ".lora_A.base_adapter.").replace(".lora_B.", ".lora_B.base_adapter.")
                adapter_state_dict[new_key] = f.get_tensor(key)
        
        logger.info("模型中的base_adapter参数名称:")
        model_param_names = [name for name, _ in model.named_parameters() if "base_adapter" in name]
        for name in model_param_names:
            logger.info(f"  {name}")
            
        logger.info("转换后的权重文件参数名称:")
        for name in adapter_state_dict.keys():
            logger.info(f"  {name}")
        
        missing_keys, unexpected_keys = model.load_state_dict(adapter_state_dict, strict=False)
        
        # 检查2: 记录加载后base_adapter参数状态
        logger.info("***检查2: 记录加载base_adapter参数后的状态")
        base_params_loaded = False
        for name, param in model.named_parameters():
            if "base_adapter" in name:
                if "lora_A" in name or "lora_B" in name:
                    log_parameter_stats(name, param)
                    if name in base_params_before:
                        param_diff = torch.abs(base_params_before[name] - param.data).mean().item()
                        if param_diff > 1e-6:
                            base_params_loaded = True
                            logger.info(f"{name} 参数加载成功，变化量: {param_diff:.8f}")
                        else:
                            logger.warning(f"警告: {name} 参数似乎没有成功加载，变化量太小: {param_diff:.8f}")
        
        if not base_params_loaded:
            logger.error("错误: base_adapter参数可能没有成功加载！")
            raise RuntimeError("base_adapter参数加载失败")
            
        logger.info(f"Missing keys: {missing_keys[:10] if len(missing_keys) > 10 else missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys[:10] if len(unexpected_keys) > 10 else unexpected_keys}")
        
        new_r = model_args.lora_r - model_args.base_lora_r
        new_config = LoraConfig(
            r = new_r,
            lora_alpha = model_args.lora_alpha,
            lora_dropout = model_args.lora_dropout,
            inference_mode = False,
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules=target_modules,
        )
        
        model.add_adapter("incremental_adapter", new_config)
        model.set_adapter(["base_adapter", "incremental_adapter"])
        
        # 检查2: 记录incremental_adapter初始化状态
        logger.info("***检查3: 检查incremental_adapter初始化状态")
        for name, param in model.named_parameters():
            if "incremental_adapter" in name:
                if "lora_A" in name:
                    log_parameter_stats(f"{name} (应该是小的随机值)", param)
                elif "lora_B" in name:
                    log_parameter_stats(f"{name} (应该初始化为0)", param)
                    is_zeros = torch.allclose(param, torch.zeros_like(param))
                    logger.info(f"{name} 是否全为0: {is_zeros}")
                    if not is_zeros:
                        logger.warning(f"警告: {name} 不是全0初始化!")
        
        # 冻结base_adapter参数
        frozen_params = 0
        total_base_params = 0
        for name, param in model.named_parameters():
            if "base_adapter" in name:
                total_base_params += param.numel()
                param.requires_grad = False
                frozen_params += param.numel()

        model.print_trainable_parameters()
        
        trainable_params = 0
        base_adapter_params = 0
        incremental_adapter_params = 0
        all_params = 0
        has_values = True
        
        sample_weight_checked = False
        sample_weight_value = None
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            
            if "base_adapter" in name:
                base_adapter_params += num_params
                if not sample_weight_checked and "lora_A" in name:
                    sample_weight_checked = True
                    sample_weight_value = param.data.flatten()[:5].tolist()
                    has_values = not torch.allclose(param.data, torch.zeros_like(param.data))
            
            if "incremental_adapter" in name:
                incremental_adapter_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
        
        logger.info(f"Active adapters: {model.active_adapters}")
        logger.info(f"Total parameters: {all_params}")
        logger.info(f"Base adapter parameters: {base_adapter_params} (frozen: {frozen_params})")
        logger.info(f"Incremental adapter parameters: {incremental_adapter_params}")
        logger.info(f"Trainable parameters: {trainable_params} ({100 * trainable_params / all_params:.2f}%)")
        logger.info(f"Base adapter loaded with values: {has_values}")
        if sample_weight_value:
            logger.info(f"Sample weights from base_adapter: {sample_weight_value}")
        
    elif model_args.incremental_lora == "domain" or model_args.incremental_lora == "all":
        config = LoraConfig(
            r = model_args.lora_r,
            lora_alpha = model_args.lora_alpha,
            lora_dropout = model_args.lora_dropout,
            inference_mode = False,
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    return model, tokenizer

def run_test(model, tokenizer, data_path, output_dir, task_type="scienceqa"):
    logger.info("开始测试模型性能...")
    
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_mc_input_short"], PROMPT_DICT["prompt_mc_no_input_short"]
    
    logger.info(f"正在加载{task_type}类型的数据集...")
    try:
        if task_type == "scienceqa":
            data = load_from_disk(data_path)
            data = data["test"]
            # Convert Dataset to list of dictionaries
            data = [dict(item) for item in data]
        else:
            with open(data_path, "r") as f:
                data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    logger.info(f"test dataset length: {len(data)}")

    acc = 0
    model.eval()
    
    results = []
    for index, example in tqdm(enumerate(data)):
        input_prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        inputs = tokenizer(input_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=10,
            )
            
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_text = full_text.removeprefix(input_prompt)
        logger.info(f"question: {input_prompt[-100:]}")
        logger.info(f"answer: {example['output']}")
        logger.info(f"model prediction: {pred_text}")
        
        example["pred"] = pred_text
        results.append(example)

        pattern = re.compile(r'The answer is ([A-Z]).')

        if task_type == "ha":
            if example["output"] in pred_text:
                acc += 1
        elif task_type == "ie":
            match = re.search(r'\[(.*?)\]', example['output'])
            if match:
                content = match.group(1).strip("'")
                if content == pred_text or pred_text.startswith(content) or content.startswith(pred_text):
                    acc += 1
        elif task_type == "scienceqa":
            match = pattern.search(example["output"])
            if match:
                answer_choice = match.group(1)
                if answer_choice in pred_text:
                    acc += 1
        elif task_type == "mc":
            if pred_text and example["output"] and pred_text[0] == example["output"][0]:
                acc += 1

        if index % 10 == 0:
            logger.info(f"Now Acc: {acc / (index+1):.4f}")

    final_acc = acc / len(data)
    logger.info(f"Final Acc: {final_acc:.4f}")

    acc_path = os.path.join(output_dir, "test_acc.json")
    with open(acc_path, "w") as f:
        json.dump({"accuracy": final_acc}, f, indent=4)

    pred_path = os.path.join(output_dir, "test_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return final_acc

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    incremental_lora = model_args.incremental_lora
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

    if model_args.incremental_lora == "task":
        monkey_patch_peft_mixed_model()

    ### Training
    logger.warning("Creating trainer...")
    
    if model_args.incremental_lora == "task":
        trainer = ParameterTrackingTrainer(
            model = model,
            tokenizer = tokenizer,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = test_dataset
        )
    else:
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
    
    logger.info("训练完成，开始测试...")
    
    task_type = "scienceqa"
    if DATASET_NAME == "ha":
        task_type = "ha"
    elif DATASET_NAME == "ie":
        task_type = "ie"
    elif DATASET_NAME == "mc":
        task_type = "mc"
    
    test_acc = run_test(
        model=model,
        tokenizer=tokenizer,
        data_path=DATASET_PATH,
        output_dir=training_args.output_dir,
        task_type=task_type
    )
    
    metrics["test_accuracy"] = test_acc
    trainer.save_metrics("test", {"accuracy": test_acc})
    
    logger.info(f"训练和测试全部完成！测试准确率: {test_acc:.4f}")

def main():
    train()

if __name__ == "__main__":
    main()
    