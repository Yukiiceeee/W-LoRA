from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Optional, Sequence, List
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
from peft import PeftModel
import json
import torch
import os
import copy
import logging
import transformers
import fire
import argparse
from tqdm import tqdm
import bert_score
import re
import numpy as np
from dataset import mcDataset
from dataset import haDataset
from constants import PROMPT_DICT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(args):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_mc_input_short"], PROMPT_DICT["prompt_mc_no_input_short"]
    model_path = args.model_path
    adapter_path = args.adapter_path
    data_path = args.data_path
    metric = args.metric
    task_type = args.task_type
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    model = PeftModel.from_pretrained(model, adapter_path)

    if task_type == "scienceqa":
        data = load_from_disk(data_path)
        data = data["test"]
    else:
        with open(data_path, "r") as f:
            data = json.load(f)
    
    logger.info(f"data length: {len(data)}")

    acc = 0
    for index, example in tqdm(enumerate(data)):
        input_prompt = prompt_input.format_map(example) if example.get("input") != "" else prompt_no_input.format_map(example)
        inputs = tokenizer(input_prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = model.generate(
            **inputs, 
            max_new_tokens=10,
            )
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        out_text = full_text.removeprefix(input_prompt)
        logger.info(f"output: {example['output']}")
        logger.info(f"pred: {out_text}")
        example["pred"] = out_text

        pattern = re.compile(r'The answer is ([A-Z]).')

        match metric:
            case "acc":
                if task_type == "ha":
                    if example["output"] in example["pred"]:
                        acc += 1
                elif task_type == "ie":
                    match = re.search(r'\[(.*?)\]', example['output'])
                    if match:
                        content = match.group(1).strip("'")
                        if content == example["pred"] or example["pred"].startswith(content) or content.startswith(example["pred"]):
                            acc += 1
                elif task_type == "scienceqa":
                    match = pattern.search(example["output"])
                    if match:
                        answer_choice = match.group(1)
                        if answer_choice in example["pred"]:
                            acc += 1

                logger.info(f"acc: {acc / (index+1)}")

    acc_path = adapter_path + "/acc.json"
    if not os.path.exists(acc_path):
        os.mknod(acc_path)
    with open(acc_path, "w") as f:
        f.write(str(acc / len(data)))

    path = adapter_path + "/pred.json"
    if not os.path.exists(path):
        os.mknod(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--adapter_path", type=str, default="/d2/mxy/W-LoRA/models/adapters/domain/fin/ha")
    parser.add_argument("--data_path", type=str, default="/d2/mxy/W-LoRA/data/Domains-Based/fin/ha/test.json")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--task_type", type=str, default="mc")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()