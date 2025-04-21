export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4

python ../src/DBSCAN_model_embed.py \
    --model_path /d2/mxy/Models/Qwen2-7B \
    --data_path /d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf \
    --output_dir /d2/mxy/W-LoRA/data/ScienceQA/DBSCAN_clustered \
    --batch_size 32 \
    --max_length 512 \
    --eps 0.5 \
    --min_samples 10 \
    --metric cosine
