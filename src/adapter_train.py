from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
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
from dataset import mcDataset
from dataset import haDataset

MyDataset = None
DATASET_PATH = None
DATASET_NAME = None
CANDIDATE_DATASETS = {
    "medmc":        [mcDataset,      "/d2/mxy/W-LoRA/data/Domains-Based/med/mc"],
    "finha":        [haDataset,      "/d2/mxy/W-LoRA/data/Domains-Based/fin/ha"],
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

    config = LoraConfig(
        r = model_args.lora_r,
        lora_alpha = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout,
        inference_mode = False,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules=get_target_modules(model.config.model_type.lower(), model.named_modules()),
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side = "left",
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )

    return model, tokenizer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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
        max_length = training_args.model_max_length
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
    