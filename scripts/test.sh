export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4

python ../src/adapter_test.py \
    --model_path /d2/mxy/Models/Qwen2-7B \
    --adapter_path /d2/mxy/W-LoRA/adapters/scienceqa \
    --data_path /d2/mxy/W-LoRA/data/ScienceQA/science_qa.hf \
    --metric acc \
    --task_type scienceqa \
    >> /d2/mxy/W-LoRA/logs/test/base_model_scienceqa.log 2>&1
