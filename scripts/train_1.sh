export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4

python ../src/adapter_train_edge_1.py \
    --use_deepspeed false \
    --model_name_or_path /d2/mxy/Models/Qwen2-7B \
    --data_name scienceqa \
    --output_dir /d2/mxy/W-LoRA/adapters/scienceqa_1 \
    --peft_type lora \
    --lora_r 10 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --incremental_lora true \
    --base_lora_path /d2/mxy/W-LoRA/adapters/scienceqa \
    --base_lora_r 8 \
    --learning_rate 2e-6 \
    --lr_scheduler_type "linear" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --model_max_length 256 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --num_train_epochs 6 \
    --logging_steps 1 \
    >> /d2/mxy/W-LoRA/logs/train_adapter_scienceqa_edge_1.log 2>&1