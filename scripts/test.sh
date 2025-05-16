export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

python ../src/adapter_test.py \
    --model_path /d2/mxy/Models/Qwen2-7B \
    --adapter_path /d2/mxy/W-LoRA/adapters/scienceqa_task \
    --data_path /d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf \
    --metric acc \
    --task_type scienceqa \
    # >> /d2/mxy/W-LoRA/logs/qwen2-7b_scienceqa_lora_task_test.log 2>&1
