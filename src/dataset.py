from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import sys
import json
import re
import os

from constants import (
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    PROMPT_DICT
)

class TaskDataset(Dataset):
    
    prompt_input = None
    prompt_no_input = None

    def __init__(self, data, tokenizer, max_length, dataset_type):
        super(TaskDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type
    
    def __len__(self):
        return len(self.data)
    
    def get_input_connected_with_label(self, idx):
        input_text = self.prompt_input.format(
            instruction = self.data[idx]["instruction"], 
            input = self.data[idx]["input"]
        ) if (self.data[idx]["input"]) else self.prompt_no_input.format(
            instruction = self.data[idx]["instruction"]
        )
        output_text = self.data[idx]["output"]
        text = input_text + output_text
        tokens = self.tokenizer(
            text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        output_tokens = self.tokenizer(
            output_text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        tokens["input_ids"] = tokens["input_ids"].flatten()
        tokens["attention_mask"] = tokens["attention_mask"].flatten()
        tokens["labels"] = output_tokens["input_ids"].flatten()
        tokens["labels"][tokens["labels"] == self.tokenizer.pad_token_id] = -100 # -100 means not counted in loss.
        return tokens
    
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        if self.dataset_type in {"train", "validation", "valid", "test"}:
            return self.get_input_connected_with_label(idx)
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        
class haDataset(TaskDataset):

    prompt_input = PROMPT_DICT["prompt_ha_input"]
    prompt_no_input = PROMPT_DICT["prompt_ha_no_input"]

    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.json"), encoding='utf-8')
        test_file = open(os.path.join(data_path, "test.json"), encoding='utf-8')
        train_data = json.load(train_file)
        test_data = json.load(test_file)
        train_file.close()
        test_file.close()

        train_dataset = mcDataset(data=train_data, tokenizer=tokenizer, max_length=max_length, dataset_type="train")
        test_dataset = mcDataset(data=test_data, tokenizer=tokenizer, max_length=max_length, dataset_type="test")
        
        return train_dataset, test_dataset

class mcDataset(TaskDataset):

    prompt_input = PROMPT_DICT["prompt_mc_input_short"]
    prompt_no_input = PROMPT_DICT["prompt_mc_no_input_short"]

    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train_processed.json"), encoding='utf-8')
        test_file = open(os.path.join(data_path, "test.json"), encoding='utf-8')
        train_data = json.load(train_file)
        test_data = json.load(test_file)
        train_file.close()
        test_file.close()

        train_dataset = mcDataset(data=train_data, tokenizer=tokenizer, max_length=max_length, dataset_type="train")
        test_dataset = mcDataset(data=test_data, tokenizer=tokenizer, max_length=max_length, dataset_type="test")
        
        return train_dataset, test_dataset
    
class MultipleChoiceQADataset(Dataset):
    MAX_SAMPLE_INPUT_LENGTH = None
    MAX_SAMPLE_OUTPUT_LENGTH = None

    prompt_template = {
        "prompt_input": None,
        "prompt_no_input": None,
    }

    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        raise NotImplementedError
    
    def __init__(self, data, tokenizer, max_length, dataset_type):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length 
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset_type in {"train", "validation", "valid", "test"}:
            return self.get_input_connected_with_label(idx)
        elif self.dataset_type == "generate":
            return self.get_input_disconnected_with_label(idx)
        else:
            raise NotImplementedError
        
    def get_input_connected_with_label(self, idx):
        input_text = self.prompt_template["prompt_input"].format(
            instruction = self.data[idx]["instruction"], 
            input = self.data[idx]["input"]
        ) if (self.data[idx]["input"]) else self.prompt_template["prompt_no_input"].format(
            instruction = self.data[idx]["instruction"]
        )
        output_text = self.data[idx]["output"]
        text = input_text + output_text
        tokens = self.tokenizer(
            text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        output_tokens = self.tokenizer(
            output_text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        tokens["input_ids"] = tokens["input_ids"].flatten()
        tokens["attention_mask"] = tokens["attention_mask"].flatten()
        tokens["labels"] = output_tokens["input_ids"].flatten()
        tokens["labels"][tokens["labels"] == self.tokenizer.pad_token_id] = -100 # -100 means not counted in loss.
        return tokens
    
    def verify_sample(self, idx):
        raw_sample = self.data[idx]
        print("\n=== 原始样本 ===")
        print(f"Instruction: {raw_sample['instruction']}")
        print(f"Input: {raw_sample['input']}")
        print(f"Output: {raw_sample['output']}")
        
        processed = self.get_input_connected_with_label(idx)
        
        input_text = self.tokenizer.decode(processed['input_ids'], skip_special_tokens=True)
        print("\n=== 处理后的输入文本 ===")
        print(input_text)
        
        valid_labels = processed['labels'][processed['labels'] != -100]
        label_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
        print("\n=== 处理后的标签文本 ===")
        print(label_text)
        
        print("\n=== 张量信息 ===")
        print(f"Input shape: {processed['input_ids'].shape}")
        print(f"Attention mask shape: {processed['attention_mask'].shape}")
        print(f"Labels shape: {processed['labels'].shape}")
        print(f"Input: {processed['input_ids']}")
        print(f"Attention mask: {processed['attention_mask']}")
        print(f"Labels: {processed['labels']}")
        
        return processed
    
class ScienceQADataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
        "response_template": r'The answer is ([A-Z]).',
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        data = load_from_disk(data_path)
        train_dataset = ScienceQADataset(data = data["train"], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = ScienceQADataset(data = data["validation"], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        return train_dataset, val_dataset

    
def main():
    print("开始验证多选题数据集...")
    tokenizer = AutoTokenizer.from_pretrained("/d2/mxy/Models/Qwen2-7B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    train_dataset, test_dataset = ScienceQADataset.load_dataset(
        data_path="/d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf",
        tokenizer=tokenizer,
        max_length=256
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    for i in range(3):
        print(f"\n检查第 {i+1} 个样本:")
        train_dataset.verify_sample(i)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(train_loader))

    for i in range(batch['input_ids'].size(0)):
        print(f"\n批次中的第 {i+1} 个样本:")
        input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
        valid_labels = batch['labels'][i][batch['labels'][i] != -100]
        label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
        print("输入:", input_text)
        print("标签:", label_text)

if __name__ == "__main__":
    main()
    