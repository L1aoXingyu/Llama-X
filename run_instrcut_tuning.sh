
set -x

deepspeed src/train_freeform.py \
    --model_name_or_path /dataset/home/liaoxingyu/models/starcoderplus \
    --data_path /dataset/home/liaoxingyu/datasets/alpaca_evol_instruct_70k.json \
    --output_dir output/starcderplus.evol_instruct_70k_v1 \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed src/configs/deepspeed_config.json \
    --fp16 True