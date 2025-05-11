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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

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

def cluster_embeddings_with_dbscan(embeddings, eps, min_samples, metric='L2'):
    logger.warning("Starting DBSCAN clustering...")
    logger.warning(f"Input embeddings shape: {embeddings.shape}")
    
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logger.warning("Warning: embeddings contain NaN or inf values!")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e20, neginf=-1e20)
    
    scaler = StandardScaler()
    try:
        embeddings_scaled = scaler.fit_transform(embeddings)
        logger.warning(f"Scaled embeddings shape: {embeddings_scaled.shape}")
        logger.warning(f"Scaled embeddings mean: {np.mean(embeddings_scaled):.4f}, std: {np.std(embeddings_scaled):.4f}")
    except Exception as e:
        logger.warning(f"Standardization failed: {str(e)}")
        return np.array([])
    
    if metric == "COS":
        norms = np.linalg.norm(embeddings_scaled, axis=1, keepdims=True)
        embeddings_scaled = embeddings_scaled / norms
        metric = "euclidean"
        logger.warning("Using normalized embeddings for cosine similarity")
    else:
        metric = "euclidean"
    
    try:
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=-1
        )
        logger.warning(f"Starting DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}")
        cluster_labels = dbscan.fit_predict(embeddings_scaled)
        logger.warning(f"DBSCAN completed. Labels shape: {cluster_labels.shape}")
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.warning(f"Number of clusters: {n_clusters}")
        logger.warning(f"Number of noise points: {n_noise}")
        
        unique_labels = np.unique(cluster_labels)
        cluster_centers_idx = []
        
        for label in unique_labels:
            if label != -1:
                cluster_points = embeddings_scaled[cluster_labels == label]
                cluster_points_idx = np.where(cluster_labels == label)[0]
                
                if len(cluster_points) > 0:
                    center = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    center_idx = cluster_points_idx[np.argmin(distances)]
                    cluster_centers_idx.append(center_idx)
                    logger.warning(f"Cluster {label}: size={len(cluster_points)}, center_idx={center_idx}")
        
        logger.warning(f"DBSCAN finished: found {len(cluster_centers_idx)} clusters")
        return np.array(cluster_centers_idx)
        
    except Exception as e:
        logger.warning(f"DBSCAN clustering failed: {str(e)}")
        return np.array([])

def selector_data_embedding(
    data_path: str,
    model_path: str,
    embedding_path: str,
    eps: float,
    min_samples: int,
    domain: str,
    task: str,
    metric: str = 'L2'
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
    labels = cluster_embeddings_with_dbscan(embeddings, eps, min_samples, metric)
    
    return [
        {**item, "domain": domain, "task": task}
        for idx, item in enumerate(data)
        if idx in labels
    ]

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--embedding_path", type=str, default="/d2/mxy/Models/Qwen2-7B/embedding.pth")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN的邻域半径参数")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN的最小样本数参数")
    parser.add_argument("--data_path", type=str, default="/d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf")
    parser.add_argument("--metric", type=str, default="COS", choices=['L2', 'COS'], help="距离度量方式")
    args = parser.parse_args()

    if args.data_path.endswith('.hf'):
        data = load_from_disk(args.data_path)
        if "train" in data:
            data = data["train"]
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    texts = [format_text(item, prompt_input, prompt_no_input) for item in data]

    embeddings = get_embeddings(texts=texts, model_id=args.model_name, embedding_id=args.embedding_path)
    labels = cluster_embeddings_with_dbscan(embeddings, args.eps, args.min_samples, args.metric)

    print(f"\nFound {len(labels)} clusters")
    print("\nCluster center indices:", labels)
    
    label_list = [0] * len(embeddings)
    for index in range(len(label_list)):
        if index in labels:
            label_list[index] = 1
    
    print(f"\nNumber of selected samples: {sum(label_list)}")
    
    print("\nSelected samples examples:")
    for i in labels[:5]:
        print(f"\nSample {i}:")
        print(texts[i])
