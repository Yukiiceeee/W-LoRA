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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
import seaborn as sns
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('sentence_transformers')
logger.setLevel(logging.WARNING)

def format_data_content(item):
    content_parts = []
    # if 'lecture' in item and item['lecture'] and str(item['lecture']).strip():
    #     lecture_content = str(item['lecture']).strip()
    #     if lecture_content.lower() not in ['none', 'null', '']:
    #         content_parts.append(lecture_content)
    if 'instruction' in item and item['instruction'] and str(item['instruction']).strip():
            instruction_content = str(item['instruction']).strip()
            if instruction_content.lower() not in ['none', 'null', '']:
                content_parts.append(instruction_content)    
    if 'input' in item and item['input'] and str(item['input']).strip():
            input_content = str(item['input']).strip()
            if input_content.lower() not in ['none', 'null', '']:
                content_parts.append(input_content)

    result = "\n".join(content_parts)

    return result

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

def preprocess_embeddings(embeddings, n_components=None, metric='cosine'):
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # PCA降维
    if n_components is not None:
        pca = PCA(n_components=n_components)
        embeddings_processed = pca.fit_transform(embeddings_normalized)
        print(f"PCA降维到{n_components}维，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
        
        # PCA后需要重新归一化（特别是对cosine距离）
        if metric == 'cosine':
            embeddings_processed = embeddings_processed / np.linalg.norm(embeddings_processed, axis=1)[:, np.newaxis]
    else:
        embeddings_processed = embeddings_normalized
    
    return embeddings_processed

def cluster_embeddings_with_dbscan(embeddings, eps=None, min_samples=5, metric='cosine'):
    logger.warning(f"Input embeddings shape: {embeddings.shape}")
    
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logger.warning("Warning: embeddings contain NaN or inf values!")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e20, neginf=-1e20)
    
    # 估计eps参数
    suggested_eps = None
    if len(embeddings) > 100:  # 降低采样阈值
        sample_size = min(1000, len(embeddings))  # 确保采样大小合理
        np.random.seed(42)
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
        neigh.fit(sample_embeddings)
        distances, _ = neigh.kneighbors()
        distances = np.sort(distances[:, -1])
        
        from kneed import KneeLocator
        kneedle = KneeLocator(range(len(distances)), distances, 
                             S=1.0, curve='convex', direction='increasing')
        if kneedle.knee is not None:
            suggested_eps = distances[kneedle.knee]
            print(f"Suggested eps based on knee point: {suggested_eps:.3f}")
            if eps is None:
                eps = suggested_eps
    if eps is None:
        eps = 0.5
    
    try:
        logger.warning(f"Starting DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}")
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=-1
        )
        cluster_labels = dbscan.fit_predict(embeddings)
        logger.warning(f"DBSCAN completed. Labels shape: {cluster_labels.shape}")

        logger.warning(f"calculating distances...")
        distances = np.full(len(cluster_labels), np.inf)
        representative_indices = []
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.warning(f"Number of clusters: {n_clusters}")
        logger.warning(f"Number of noise points: {n_noise}")
        
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label != -1:
                mask = cluster_labels == label
                cluster_points = embeddings[mask]
                cluster_indices = np.where(mask)[0]

                if(len(cluster_indices) > 0):
                    center = np.mean(cluster_points, axis=0)
                    if metric == "cosine":
                        normalized_center = center / np.linalg.norm(center)
                        normalized_points = cluster_points / np.linalg.norm(cluster_points, axis=1, keepdims=True)
                        cosine_similarities = np.dot(normalized_points, normalized_center)
                        cluster_distances = 1 - cosine_similarities
                    else:  # euclidean
                        cluster_distances = np.linalg.norm(cluster_points - center, axis=1)
                        
                distances[cluster_indices] = cluster_distances

                n_representatives = max(1, int(len(cluster_indices) * 0.5))
                closest_indices = cluster_indices[np.argsort(cluster_distances)[:n_representatives]]
                representative_indices.extend(closest_indices.tolist())
        
                logger.warning(f"Cluster {label}:")
                logger.warning(f"Size: {len(cluster_indices)}")
                logger.warning(f"Selected points: {n_representatives}")
                logger.warning(f"Min distance: {np.min(cluster_distances):.4f}")
                logger.warning(f"Max distance: {np.max(cluster_distances):.4f}")
                logger.warning(f"Mean distance: {np.mean(cluster_distances):.4f}")

        if -1 in cluster_labels:
            max_valid_distance = np.max(distances[distances != np.inf])
            distances[cluster_labels == -1] = max_valid_distance * 1.5
            logger.warning(f"Noise points: {n_noise}, assigned distance: {max_valid_distance * 1.5:.4f}")
                
        logger.warning(f"DBSCAN finished: found {n_clusters} clusters, "
                      f"selected {len(representative_indices)} representative points")
        return cluster_labels, np.array(representative_indices), distances
        
    except Exception as e:
        logger.warning(f"DBSCAN clustering failed: {str(e)}")
        return np.array([]), np.array([]), np.array([])

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
    labels, representative_indices, distances = cluster_embeddings_with_dbscan(embeddings, eps, min_samples, metric)
    
    return [
        {**item, "domain": domain, "task": task}
        for idx, item in enumerate(data)
        if idx in labels
    ]

def visualize_clusters(embeddings, labels, distances=None, save_path='cluster_visualization.png', 
                      method='tsne', title_suffix='', metric='cosine'):
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method.lower() == 'tsne':
        if metric == 'cosine':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, metric='cosine')
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, metric='euclidean')
    else:  # umap
        if metric == 'cosine':
            reducer = UMAP(random_state=42, metric='cosine')
        else:
            reducer = UMAP(random_state=42, metric='euclidean')
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(15, 10))
    
    if distances is not None:
        valid_distances = distances[distances != np.inf]
        if len(valid_distances) > 0:
            min_dist, max_dist = np.min(valid_distances), np.max(valid_distances)
            if max_dist > min_dist:
                sizes = 20 + 180 * (1 - (distances - min_dist) / (max_dist - min_dist))
                sizes[distances == np.inf] = 10
            else:
                sizes = np.full(len(distances), 50)
        else:
            sizes = np.full(len(distances), 50)
    else:
        sizes = np.full(len(labels), 50)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    scatter_objects = []
    
    for i, (label, color) in enumerate(zip(unique_labels, colors)):
        if label == -1:
            mask = labels == -1
            scatter = plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c='gray', marker='x', alpha=0.5, s=20, label='Noise')
        else:
            mask = labels == label
            scatter = plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[color], alpha=0.7, s=sizes[mask], 
                       label=f'Cluster {label} (n={np.sum(mask)})')
            scatter_objects.append(scatter)
    
    for label in unique_labels:
        if label != -1:
            mask = labels == label
            if np.any(mask):
                center_2d = np.mean(embeddings_2d[mask], axis=0)
                plt.scatter(center_2d[0], center_2d[1], 
                           c='red', marker='*', s=200, alpha=0.8,
                           edgecolors='black', linewidth=1)
                
                if distances is not None:
                    avg_dist = np.mean(distances[mask])
                    plt.annotate(f'C{label}\nAvg: {avg_dist:.3f}', 
                               xy=(center_2d[0], center_2d[1]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
                               fontsize=8)
    
    plt.title(f'{method.upper()} Visualization ({metric.upper()} metric)\n'
              f'Point size ∝ proximity to center{title_suffix}')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")
    
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
    print(f"聚类统计: {n_clusters}个聚类, {n_noise}个噪声点")
    
    return embeddings_2d

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--embedding_path", type=str, default="/d2/mxy/Models/Qwen2-7B/embedding.pth")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN的邻域半径参数，None表示自动估计")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN的最小样本数参数")
    parser.add_argument("--data_path", type=str, default="/d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf")
    parser.add_argument("--metric", type=str, default="cosine", choices=['euclidean', 'cosine'], help="距离度量方式")
    parser.add_argument("--pca_components", type=int, default=128, help="PCA的组件数，更高维度可能保留更多信息")
    parser.add_argument("--task_type", type=str, default="qa", help="任务类型")
    parser.add_argument("--viz_method", type=str, default="both", choices=["tsne", "umap", "both"], help="可视化方法")
    parser.add_argument("--output_path", type=str, default="/d2/mxy/W-LoRA/src/cluster_output")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if args.data_path.endswith('.hf'):
        data = load_from_disk(args.data_path)
        if "train" in data:
            data = data["train"]
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)

    texts = [format_data_content(item) for item in data]

    print("Getting embeddings...")
    embeddings = get_embeddings(texts=texts, model_id=args.model_name, embedding_id=args.embedding_path)

    print("Preprocessing embeddings...")
    n_components = args.pca_components if args.pca_components > 0 else None
    embeddings_processed = preprocess_embeddings(embeddings, n_components=n_components, metric=args.metric)

    print("Clustering embeddings...")
    labels, representative_indices, distances = cluster_embeddings_with_dbscan(embeddings_processed, args.eps, args.min_samples, args.metric)

    # === 验证功能 1: 距离选择验证 ===
    print(f"\n=== 距离选择验证 ===")
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1:  # 跳过噪声点
            mask = labels == label
            cluster_indices = np.where(mask)[0]
            cluster_distances = distances[mask]
            
            n_representatives = max(1, int(len(cluster_indices) * 0.5))
            sorted_indices = np.argsort(cluster_distances)
            
            selected_distances = cluster_distances[sorted_indices[:n_representatives]]
            unselected_distances = cluster_distances[sorted_indices[n_representatives:]]
            
            print(f"\nCluster {label} (总共{len(cluster_indices)}个点):")
            print(f"  选中的{n_representatives}个点距离: min={np.min(selected_distances):.4f}, max={np.max(selected_distances):.4f}, mean={np.mean(selected_distances):.4f}")
            if len(unselected_distances) > 0:
                print(f"  未选中的{len(unselected_distances)}个点距离: min={np.min(unselected_distances):.4f}, max={np.max(unselected_distances):.4f}, mean={np.mean(unselected_distances):.4f}")
            print(f"  验证: 选中点最大距离 {'<=' if len(unselected_distances) == 0 or np.max(selected_distances) <= np.min(unselected_distances) else '>'} 未选中点最小距离")

    # === 验证功能 2: 样本内容验证 ===
    print(f"\n=== 样本内容验证 ===")
    selected_samples = set(representative_indices)
    
    for label in unique_labels:
        if label != -1:  # 跳过噪声点
            mask = labels == label
            cluster_indices = np.where(mask)[0]
            
            # 找到这个簇中被选中的前3个样本
            cluster_selected = [idx for idx in cluster_indices if idx in selected_samples][:3]
            
            print(f"\nCluster {label} - 前3个代表样本:")
            for i, idx in enumerate(cluster_selected):
                sample_text = texts[idx][:150] + "..." if len(texts[idx]) > 150 else texts[idx]
                print(f"  样本{i+1} (距离质心: {distances[idx]:.4f}):")
                print(f"    {sample_text}")
                print()

    print("Visualizing clusters...")
    if args.viz_method in ['tsne', 'both']:
        vis_path = os.path.join(args.output_path, 'cluster_visualization_tsne.png')
        visualize_clusters(embeddings_processed, labels, distances, save_path=vis_path, 
                         method='tsne', title_suffix=f'\neps={args.eps}, min_samples={args.min_samples}', metric=args.metric)
    if args.viz_method in ['umap', 'both']:
        vis_path = os.path.join(args.output_path, 'cluster_visualization_umap.png')
        visualize_clusters(embeddings_processed, labels, distances, save_path=vis_path, 
                         method='umap', title_suffix=f'\neps={args.eps}, min_samples={args.min_samples}', metric=args.metric)
    
    # 修复bug: 使用representative_indices来标记选中的样本
    selected_samples = set(representative_indices)
    label_list = [1 if i in selected_samples else 0 for i in range(len(embeddings))]
    
    print(f"\nNumber of selected samples: {sum(label_list)}")
    print(f"Total samples: {len(embeddings)}")
    print(f"Selection ratio: {sum(label_list)/len(embeddings):.3f}")
    
    print("\nSelected samples examples:")
    for i, idx in enumerate(representative_indices[:5]):
        print(f"\nSample {idx} (Cluster {labels[idx]}):")
        print(f"Distance to center: {distances[idx]:.4f}")
        print(texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx])
