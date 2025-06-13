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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from umap.umap_ import UMAP
import seaborn as sns
from Kmeans_model_embed import get_embeddings, format_text

def format_data_content(item):
    """
    直接格式化数据集的核心内容（lecture、question、answer），不使用prompt模板
    专门针对ScienceQA等问答数据集设计
    """
    content_parts = []
    
    if 'lecture' in item and item['lecture'] and str(item['lecture']).strip():
        lecture_content = str(item['lecture']).strip()
        if lecture_content.lower() not in ['none', 'null', '']:
            content_parts.append(f"讲座内容: {lecture_content}")
    
    if 'question' in item and item['question'] and str(item['question']).strip():
        question_content = str(item['question']).strip()
        if question_content.lower() not in ['none', 'null', '']:
            content_parts.append(f"问题: {question_content}")
    
    if 'answer' in item and item['answer'] and str(item['answer']).strip():
        answer_content = str(item['answer']).strip()
        if answer_content.lower() not in ['none', 'null', '']:
            content_parts.append(f"答案: {answer_content}")
    
    if 'solution' in item and item['solution'] and str(item['solution']).strip():
        solution_content = str(item['solution']).strip()
        if solution_content.lower() not in ['none', 'null', '']:
            content_parts.append(f"解答: {solution_content}")
    
    if not content_parts:
        if 'instruction' in item and item['instruction'] and str(item['instruction']).strip():
            instruction_content = str(item['instruction']).strip()
            if instruction_content.lower() not in ['none', 'null', '']:
                content_parts.append(f"指令: {instruction_content}")
        
        if 'input' in item and item['input'] and str(item['input']).strip():
            input_content = str(item['input']).strip()
            if input_content.lower() not in ['none', 'null', '']:
                content_parts.append(f"输入: {input_content}")
    
    result = "\n".join(content_parts) if content_parts else "空内容"
    
    if not content_parts:
        print(f"警告: 数据项缺少内容，可用字段: {list(item.keys())}")
    
    return result

def preprocess_embeddings(embeddings, n_components=None):
    """
    预处理嵌入向量：标准化，可选PCA降维
    """
    # 首先进行L2归一化
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # 标准化
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_normalized)
    
    # PCA降维（可选）
    if n_components is not None:
        pca = PCA(n_components=n_components)
        embeddings_scaled = pca.fit_transform(embeddings_scaled)
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return embeddings_scaled

def cluster_with_dbscan(embeddings, eps=0.5, min_samples=5, metric='cosine'):
    """
    使用DBSCAN进行聚类
    """
    print("Performing DBSCAN clustering...")
    
    # 首先计算一个较小样本的eps建议值
    if len(embeddings) > 1000:
        sample_size = 1000
        np.random.seed(42)
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
        neigh.fit(sample_embeddings)
        distances, _ = neigh.kneighbors()
        distances = np.sort(distances[:, -1])
        
        # 找到距离分布的拐点作为eps建议值
        from kneed import KneeLocator
        kneedle = KneeLocator(range(len(distances)), distances, 
                             S=1.0, curve='convex', direction='increasing')
        if kneedle.knee is not None:
            suggested_eps = distances[kneedle.knee]
            print(f"Suggested eps based on knee point: {suggested_eps:.3f}")
            if eps is None:
                eps = suggested_eps
    
    if eps is None:
        eps = 0.5  # 默认值
    
    print(f"Using eps={eps}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = dbscan.fit_predict(embeddings)
    
    # 计算每个点到其簇中心的距离
    distances = np.zeros(len(labels))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1:  # -1表示噪声点
            mask = labels == label
            cluster_points = embeddings[mask]
            center = np.mean(cluster_points, axis=0)
            
            if metric == 'cosine':
                # 计算余弦距离
                norm_center = center / np.linalg.norm(center)
                norm_points = cluster_points / np.linalg.norm(cluster_points, axis=1)[:, np.newaxis]
                dist = 1 - np.dot(norm_points, norm_center)
            else:
                # 计算欧氏距离
                dist = np.linalg.norm(cluster_points - center, axis=1)
                
            distances[mask] = dist
    
    # 将噪声点的距离设为最大值
    if np.any(labels == -1):
        distances[labels == -1] = np.max(distances[labels != -1])
    
    # 如果噪声点太多，尝试重新聚类
    noise_ratio = np.sum(labels == -1) / len(labels)
    if noise_ratio > 0.5:  # 如果噪声点超过50%
        print(f"Warning: {noise_ratio*100:.1f}% points are noise. Trying larger eps...")
        return cluster_with_dbscan(embeddings, eps=eps*1.5, min_samples=min_samples, metric=metric)
    
    return labels, distances

def visualize_clusters(embeddings, labels, distances=None, save_path='cluster_visualization.png', 
                      method='tsne', title_suffix=''):
    """
    使用t-SNE或UMAP将高维嵌入降到2维并可视化聚类结果
    """
    # 降维
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    else:  # umap
        reducer = UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 创建散点图
    plt.figure(figsize=(15, 10))
    
    # 如果有距离信息，用它来调整点的大小
    if distances is not None:
        # 将距离归一化到[20, 200]范围内作为点的大小
        sizes = 20 + 180 * (1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances)))
    else:
        sizes = 50
    
    # 处理噪声点（标签为-1的点）
    if -1 in labels:
        # 先画噪声点
        noise_mask = labels == -1
        plt.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                   c='gray', marker='x', alpha=0.5, s=50, label='Noise')
        # 再画聚类点
        scatter = plt.scatter(embeddings_2d[~noise_mask, 0], embeddings_2d[~noise_mask, 1],
                            c=labels[~noise_mask], cmap='tab10', alpha=0.6, s=sizes[~noise_mask])
    else:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=sizes)
    
    # 添加颜色条和标题
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'{method.upper()} Visualization of Clusters\n(Point size indicates proximity to cluster center){title_suffix}')
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    
    # 计算和显示聚类中心
    unique_labels = np.unique(labels)
    for i in unique_labels:
        if i != -1:  # 跳过噪声点
            mask = labels == i
            if np.any(mask):
                center = np.mean(embeddings_2d[mask], axis=0)
                if distances is not None:
                    avg_dist = np.mean(distances[mask])
                    plt.annotate(f'Cluster {i}\nAvg dist: {avg_dist:.3f}\nSize: {np.sum(mask)}', 
                               xy=(center[0], center[1]),
                               xytext=(5, 5), textcoords='offset points',
                               bbox=dict(facecolor='white', alpha=0.7))
                else:
                    plt.annotate(f'Cluster {i}\nSize: {np.sum(mask)}', 
                               xy=(center[0], center[1]),
                               xytext=(5, 5), textcoords='offset points',
                               bbox=dict(facecolor='white', alpha=0.7))
    
    if -1 in labels:
        plt.legend()
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/d2/mxy/Models/Qwen2-7B")
    parser.add_argument("--embedding_path", type=str, default="/d2/mxy/Models/Qwen2-7B/embedding.pth")
    parser.add_argument("--data_path", type=str, default="/d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf")
    parser.add_argument("--task_type", type=str, default="qa")
    parser.add_argument("--output_dir", type=str, default="visualization_output")
    parser.add_argument("--pca_components", type=int, default=64, help="Number of PCA components, 0 for no PCA")
    parser.add_argument("--eps", type=float, default=0.31, help="DBSCAN的邻域半径，None为自动确定")
    parser.add_argument("--min_samples", type=int, default=10, help="DBSCAN的最小样本数")
    parser.add_argument("--viz_method", type=str, default="both", choices=['tsne', 'umap', 'both'], 
                       help="选择可视化方法")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    if args.data_path.endswith('.hf'):
        data = load_from_disk(args.data_path)
        if "train" in data:
            data = data["train"]
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)

    texts = [format_data_content(item) for item in data]

    # 获取嵌入
    print("Getting embeddings...")
    embeddings = get_embeddings(texts=texts, model_id=args.model_name, embedding_id=args.embedding_path)
    
    # 预处理嵌入
    print("Preprocessing embeddings...")
    n_components = args.pca_components if args.pca_components > 0 else None
    embeddings_processed = preprocess_embeddings(embeddings, n_components=n_components)
    
    # 聚类
    labels, distances = cluster_with_dbscan(embeddings_processed, eps=args.eps, 
                                          min_samples=args.min_samples, metric='cosine')

    # 可视化结果
    print("Visualizing clusters...")
    if args.viz_method in ['tsne', 'both']:
        vis_path = os.path.join(args.output_dir, 'cluster_visualization_tsne.png')
        visualize_clusters(embeddings_processed, labels, distances, save_path=vis_path, 
                         method='tsne', title_suffix=f'\neps={args.eps}, min_samples={args.min_samples}')
    
    if args.viz_method in ['umap', 'both']:
        vis_path = os.path.join(args.output_dir, 'cluster_visualization_umap.png')
        visualize_clusters(embeddings_processed, labels, distances, save_path=vis_path, 
                         method='umap', title_suffix=f'\neps={args.eps}, min_samples={args.min_samples}')

    # 打印聚类统计信息
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
    
    print(f"\nDBSCAN Clustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    
    print("\nCluster statistics:")
    for label in unique_labels:
        if label != -1:
            mask = labels == label
            cluster_size = np.sum(mask)
            avg_distance = np.mean(distances[mask])
            print(f"\nCluster {label}: {cluster_size} samples ({cluster_size/len(labels)*100:.2f}%), "
                  f"Average distance: {avg_distance:.4f}")
            
            # 找出该簇中距离最近和最远的样本
            cluster_distances = distances[mask]
            cluster_indices = np.where(mask)[0]
            nearest_idx = cluster_indices[np.argmin(cluster_distances)]
            farthest_idx = cluster_indices[np.argmax(cluster_distances)]
            
            print(f"\n  Nearest sample (distance: {distances[nearest_idx]:.4f}):")
            print(f"  {texts[nearest_idx][:300]}...")
            print(f"\n  Farthest sample (distance: {distances[farthest_idx]:.4f}):")
            print(f"  {texts[farthest_idx][:300]}...")
            print("\n" + "-"*80)

if __name__ == "__main__":
    main() 