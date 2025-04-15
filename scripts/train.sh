export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

python ../src/adapter_train.py \
    --use_deepspeed false \
    --model_name_or_path /d2/mxy/Models/Qwen2-7B \
    --data_name scienceqa \
    --output_dir /d2/mxy/W-LoRA/adapters/scienceqa \
    --peft_type lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
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
    >> /d2/mxy/W-LoRA/logs/train/train_adapter_scienceqa.log 2>&1