import json
import random
import os
from typing import List, Dict
import argparse

def load_json(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def split_dataset(data: List[Dict], val_ratio: float = 0.1, seed: int = 42) -> tuple[List[Dict], List[Dict]]:
    """将数据集分割为训练集和验证集"""
    random.seed(seed)
    # 复制数据以避免修改原始数据
    data_copy = data.copy()
    # 随机打乱数据
    random.shuffle(data_copy)
    
    # 计算验证集大小
    val_size = int(len(data_copy) * val_ratio)
    
    # 分割数据
    val_data = data_copy[:val_size]
    train_data = data_copy[val_size:]
    
    return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train and validation sets')
    parser.add_argument('--input_file', type=str, default="/d2/mxy/W-LoRA/data/Domains-Based/med/mc/train.json", help='Input JSON file path')
    parser.add_argument('--output_dir', type=str, default="/d2/mxy/W-LoRA/data/Domains-Based/med/mc", help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples from {args.input_file}")
    
    # 分割数据集
    train_data, val_data = split_dataset(data, args.val_ratio, args.seed)
    
    # 构建输出文件路径
    train_output = os.path.join(args.output_dir, 'train2.json')
    val_output = os.path.join(args.output_dir, 'test2.json')
    
    # 保存数据集
    save_json(train_data, train_output)
    save_json(val_data, val_output)
    
    print(f"Split complete!")
    print(f"Training set: {len(train_data)} samples -> {train_output}")
    print(f"Validation set: {len(val_data)} samples -> {val_output}")

if __name__ == '__main__':
    main()