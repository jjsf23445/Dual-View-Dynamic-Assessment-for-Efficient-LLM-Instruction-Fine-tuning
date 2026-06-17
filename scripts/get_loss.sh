cd ..

python data_selection/get_loss_entropy.py \
    --model_name_or_path models/skyline2006/llama-7b \
    --data_path datasets/alpaca_data.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --tf32 True