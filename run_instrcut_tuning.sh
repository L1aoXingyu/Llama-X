
set -x

# <data>-gpt-<#params>-<precision>-<arch>-<#bsz>-<#ctxlen>-<#tok>-<#nodes>-<cluster-name>-<etc>
RUN_NAME=alpaca_evol_v1_code_evol_v1-starcoderplus-15b-fp16-zero_dp-plr2e-5-mlr0-mbsz16-gbsz512-ctxlen2048-tokn90k_piece-ep3-wmup30

export WANDB_PROJECT=WizardCoder

deepspeed src/train_freeform.py \
    --model_name_or_path /dataset/home/liaoxingyu/models/starcoderplus \
    --data_path \
    /dataset/home/liaoxingyu/datasets/alpaca_evol_instruct_70k.json \
    /dataset/home/liaoxingyu/datasets/code_evol_20k_v1.json \
    --output_dir output/$RUN_NAME \
    --run_name $RUN_NAME \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \
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
    --report_to "all" \
    --gradient_checkpointing True \
    --deepspeed src/configs/deepspeed_config.json \
    --fp16 True