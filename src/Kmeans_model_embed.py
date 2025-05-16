import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import os
from layers import separate_embedding_layer, load_embedding_layer
from datasets import load_from_disk
import logging
import json
import argparse
from constants import PROMPT_DICT

logger = logging.getLogger('sentence_transformers')
logger.setLevel(logging.WARNING)

def load_model_and_tokenizer(modelpath: str, embeddingpath: str, ):
    model = load_embedding_layer(embeddingpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
    
def format_text(item, prompt_input, prompt_no_input):
    if item.get("input"):
        return prompt_input.format(
            instruction=item["instruction"],
            input=item["input"]
        )
    else:
        return prompt_no_input.format(
            instruction=item["instruction"]
        )
    
def get_embeddings(texts, model_id, embedding_id):
    model, tokenizer = load_model_and_tokenizer(model_id, embedding_id)
    
    model.eval()
    
    # inputs = [tokenizer(text, return_tensors="pt", padding=True, truncation=True).to('cuda') for text in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to('cuda:0')
    inputs_ids = inputs['input_ids']

    output_averages = []

    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(inputs_ids):
            output = model(i)
            output_average = torch.mean(output, dim=0)
            output_averages.append(output_average.cpu().numpy())
    output_averages = np.array(output_averages).squeeze()

    del model
    
    return output_averages

def cluster_embeddings_with_faiss(embeddings, n_clusters, metric):
    logging.info("Kmeans...")
    # 转换为FAISS的float32格式
    embeddings_faiss = embeddings.astype(np.float32)
    # embeddings_faiss = embeddings
    # 创建FAISS索引
    d = embeddings_faiss.shape[1]
    if metric == "L2":
        logger.warning("Using L2 distance")
        index = faiss.IndexFlatL2(d)
    elif metric == "IP":
        logger.warning("Using IP distance")
        index = faiss.IndexFlatIP(d)
    elif metric == "COS":
        # Normalization plus inner product equals cosine similarity
        norms = np.linalg.norm(embeddings_faiss, axis=1, keepdims=True)
        embeddings_faiss = embeddings_faiss / norms   
        logger.warning("Using cosine distance")
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError("Invalid distance metric")

    # 添加数据到索引
    index.add(embeddings_faiss)

    # 进行K-means聚类
    kmeans = faiss.Kmeans(d, n_clusters, verbose=True)
    kmeans.train(embeddings_faiss)

    # 为每个样本分配聚类标签和获取到质心的距离
    distances, labels = kmeans.index.search(embeddings_faiss, 1)
    
    logger.warning(f"Kmeans finished:{len(labels.flatten())}")
    
    # 如果使用的是内积相似度（IP或COS），将距离转换为相似度
    if metric in ["IP", "COS"]:
        # 内积距离越大表示越相似，我们用1减去归一化后的距离得到相似度
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        distances = 1 - (distances - min_dist) / (max_dist - min_dist)

    return labels.flatten(), distances.flatten()

def selector_data_embedding(
    data_path: str,
    model_path: str,
    embedding_path: str,
    n_clusters: int,
    domain: str,
    task: str,
    metric: str
):
    if data_path.endswith('.hf'):
        data = load_from_disk(data_path)
        if isinstance(data, dict):
            data = data["train"]
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    prompt_input = PROMPT_DICT["prompt_input"]
    prompt_no_input = PROMPT_DICT["prompt_no_input"]
    
    texts = [format_text(item, prompt_input, prompt_no_input) for item in data]

    embeddings = get_embeddings(texts=texts, model_id=model_path, embedding_id=embedding_path)
    labels, distances = cluster_embeddings_with_faiss(embeddings, n_clusters, metric)
    
    # 返回带有聚类信息的数据
    return [
        {**item, "domain": domain, "task": task, "cluster_distance": float(dist)}
        for idx, (item, dist) in enumerate(zip(data, distances))
        if idx in labels
    ]

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--embedding_path", type=str, default="/d2/mxy/Models/Qwen2-7B/embedding.pth")
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="/d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf")
    parser.add_argument("--task_type", type=str, default="qa")
    parser.add_argument("--metric", type=str, default="L2")
    args = parser.parse_args()

    if args.data_path.endswith('.hf'):  # HuggingFace dataset
        data = load_from_disk(args.data_path)
        if "train" in data:
            data = data["train"]
    else:  # JSON file
        with open(args.data_path, 'r') as f:
            data = json.load(f)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    texts = [format_text(item, prompt_input, prompt_no_input) for item in data]

    embeddings = get_embeddings(texts=texts, model_id=args.model_name, embedding_id=args.embedding_path)
    labels, distances = cluster_embeddings_with_faiss(embeddings, args.n_clusters, args.metric)

    # 打印每个簇的样本数量和平均距离
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        cluster_size = np.sum(mask)
        avg_distance = np.mean(distances[mask])
        print(f"簇 {label}: {cluster_size} 个样本, 平均距离: {avg_distance:.4f}")
    
    # 找出每个簇中距离质心最近和最远的样本
    for label in unique_labels:
        mask = labels == label
        cluster_distances = distances[mask]
        cluster_indices = np.where(mask)[0]
        
        nearest_idx = cluster_indices[np.argmin(cluster_distances)]
        farthest_idx = cluster_indices[np.argmax(cluster_distances)]
        
        print(f"\n簇 {label}:")
        print(f"最近的样本 (距离: {distances[nearest_idx]:.4f}):")
        print(texts[nearest_idx][:200] + "...")  # 只打印前200个字符
        print(f"最远的样本 (距离: {distances[farthest_idx]:.4f}):")
        print(texts[farthest_idx][:200] + "...")
