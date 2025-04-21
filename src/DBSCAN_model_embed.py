import numpy as np
import faiss
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from constants import PROMPT_DICT
from sklearn.cluster import DBSCAN
import fire
import datasets
from datasets import Dataset
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import sys
from datasets import load_from_disk
import logging
import argparse
from collections import Counter
from constants import PROMPT_DICT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(modelpath: str):
    model = AutoModelForCausalLM.from_pretrained(
        modelpath,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        modelpath, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_embeddings(texts, args):
    model_path = args.model_path
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to('cuda:0')
    model.eval()

    model.config.output_hidden_states = True
    
    output_averages = []
    batch_size = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                padding_side="left",
                max_length=args.max_length,
                return_tensors="pt"
            ).to('cuda:0')
            outputs = model(**inputs, output_hidden_states=True)
            if outputs.hidden_states is None:
                raise ValueError("Model did not output hidden states. Please check model configuration.")
            last_hidden_states = outputs.hidden_states[-1]
            masks = inputs["attention_mask"].unsqueeze(-1)
            masked_states = last_hidden_states * masks
            sum_states = masked_states.sum(dim=1)
            counts = masks.sum(dim=1)
            counts = torch.clamp(counts, min=1e-9)
            avg_states = (sum_states / counts).cpu().numpy()
            output_averages.extend(avg_states)
    
    del model
    torch.cuda.empty_cache()

    output_array = np.stack(output_averages, axis=0)
    logger.info(f"输出数组形状: {output_array.shape}")
    logger.info(f"输出数组前5行: {output_array[:5]}")
    return output_array

def reduce_dim(embeddings, args):
    logger.info(f"原始嵌入形状: {embeddings.shape}")
    # scaler = StandardScaler()
    # scaled_embeddings = scaler.fit_transform(embeddings)
    scaled_embeddings = embeddings

    reducer = umap.UMAP(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=42,
        metric=args.metric
    )

    logger.info("使用UMAP进行降维...")
    reduced_embeddings = reducer.fit_transform(scaled_embeddings)
    logger.info(f"降维后维度: {reduced_embeddings.shape}")
    logger.info(f"降维后前5行: {reduced_embeddings[:5]}")

    return reduced_embeddings

def dbscan(embeddings, args):
    eps = args.eps
    min_samples = args.min_samples
    metric = args.metric
    
    # norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # normalized_embeddings = embeddings / norms
    normalized_embeddings = embeddings

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=-1
    )
    cluster_labels = db.fit_predict(normalized_embeddings)

    label_counts = Counter(cluster_labels)

    largest_cluster = max(label_counts.items(), key=lambda x: x[1])[0]
    
    return cluster_labels, largest_cluster

def process(args):
    dataset = load_from_disk(args.data_path)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_dataset = train_dataset.select(range(1000))

    logger.info(f"训练集样本数量: {len(train_dataset)}")
    logger.info(f"验证集样本数量: {len(val_dataset)}")

    texts = []
    for item in train_dataset:
        prompt = PROMPT_DICT["prompt_mc_input_short"].format(
            instruction = item["instruction"],
            input = item["input"] if item["input"] != "" else ""
        )
        texts.append(prompt)
    
    logger.info("获取训练集embedding...")
    embeddings = get_embeddings(texts, args)
    logger.info("进行降维处理...")
    reduced_embeddings = reduce_dim(embeddings, args)
    logger.info("进行DBSCAN聚类...")
    cluster_labels, largest_cluster = dbscan(reduced_embeddings, args)

    largest_cluster_indices = [int(i) for i in np.where(cluster_labels == largest_cluster)[0]]
    other_indices = [int(i) for i in np.where(cluster_labels != largest_cluster)[0]]

    logger.info(f"簇的数量：{len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    logger.info(f"最大簇的样本数量: {len(largest_cluster_indices)}")
    logger.info(f"其他簇的样本数量: {len(other_indices)}")

    largest_cluster_data = [train_dataset[i] for i in largest_cluster_indices]
    other_cluster_data = [train_dataset[i] for i in other_indices]
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    largest_cluster_dataset = Dataset.from_list(largest_cluster_data)
    other_cluster_dataset = Dataset.from_list(other_cluster_data)

    # largest_cluster_dataset.save_to_disk(os.path.join(output_dir, "largest_cluster"))
    # other_cluster_dataset.save_to_disk(os.path.join(output_dir, "other_clusters"))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--data_path", type=str, default="/d2/mxy/W-LoRA/data/Domains-Based/fin/ha/test.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eps", type=float, default=100)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--output_dir", type=str, default="/d2/mxy/W-LoRA/data/ScienceQA/DBSCAN_clustered")
    parser.add_argument("--n_components", type=int, default=100, help="降维后的维度")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP邻居数量")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP最小距离")
    args = parser.parse_args()
    
    process(args)

if __name__ == "__main__":
    main()
