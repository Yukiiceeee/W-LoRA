export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4

# python ../src/DBSCAN_model_embed.py \
#     --model_name /d2/mxy/Models/Qwen2-7B \
#     --embedding_path /d2/mxy/Models/Qwen2-7B/embedding.pth \
#     --eps 0.2 \
#     --min_samples 3 \
#     --data_path /d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf \
#     --metric cosine \
#     --pca_components 64 \
#     --task_type qa \
#     --viz_method both \
#     --output_path /d2/mxy/W-LoRA/src/cluster_output \
#     >> /d2/mxy/W-LoRA/logs/qwen2_7b_scienceqa_DBSCAN.log 2>&1

python ../src/KMeans_model_embed.py \
    --model_name /d2/mxy/Models/Qwen2-7B \
    --embedding_path /d2/mxy/Models/Qwen2-7B/embedding.pth \
    --n_clusters 24 \
    --max_iter 300 \
    --data_path /d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf \
    --metric cosine \
    --pca_components 64 \
    --task_type qa \
    --viz_method both \
    --output_path /d2/mxy/W-LoRA/src/cluster_output \
    >> /d2/mxy/W-LoRA/logs/qwen2_7b_scienceqa_KMeans.log 2>&1