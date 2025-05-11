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
    # niter：迭代次数，verbose：log全部输出
    # kmeans = faiss.Kmeans(d, n_clusters, niter=50, verbose=True)
    kmeans = faiss.Kmeans(d, n_clusters, verbose=True)
    kmeans.train(embeddings_faiss)

    # 如下是返回每个数据样本属于哪个簇
    # _, labels = kmeans.index.search(embeddings_faiss, 1)
    # 如下是返回对应离每个质心最近的数据样本的索引序号
    _, labels = index.search(kmeans.centroids, 1)
    
    logger.warning(f"Kmeans finished:{len(labels.flatten())}")

    return labels.flatten()

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
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters, metric)
    
    return [
        {**item, "domain": domain, "task": task}
        for idx, item in enumerate(data)
        if idx in labels
    ]

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--embedding_path", type=str, default="/d2/mxy/Models/Qwen2-7B/embedding.pth")
    parser.add_argument("--n_clusters", type=int, default=100)
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
    labels = cluster_embeddings_with_faiss(embeddings, args.n_clusters, args.metric)

    for i in labels:
        print(f"聚类标签: {i}")
    label_list = [0] * len(embeddings)
    for index, label in enumerate(label_list):
        if index in labels:
            label_list[index] = 1
    print(sum(label_list))
    # for i in labels:
    #     print(f"聚类中心: {texts[i]}")
    # for i, text in enumerate(texts):
    #     print(f"文本: {texts}, 聚类标签: {labels[i]}")
