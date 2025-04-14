export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

python ../src/adapter_test.py \
    --model_path /d2/mxy/Models/Qwen2-7B \
    --adapter_path /d2/mxy/W-LoRA/adapters/Qwen2-7B/medmc \
    --data_path /d2/mxy/W-LoRA/data/Domains-Based/med/mc/test.json \
    --metric acc \
    --task_type ha \
    >> /d2/mxy/W-LoRA/logs/test/base_model_med_mc.log 2>&1
