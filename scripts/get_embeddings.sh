#!/bin/bash
cd ..
train_file="datasets/alpaca_data.json"
warm_model="models/warmup/checkpoint-204"
base_model="models/skyline2006/llama-7b"
grad_output_path="save/alpaca/grad"
semantic_output_path="save/alpaca/semantic"
dims="4096"
gradient_type="adam"

for path in "$grad_output_path" "$semantic_output_path"; do
    if [[ ! -d $path ]]; then
        mkdir -p $path
    fi
done
    
python3 -m data_selection.get_info \
    --train_file "$train_file" \
    --info_type grads \
    --model_path "$warm_model" \
    --output_path "$grad_output_path" \
    --gradient_projection_dimension "$dims" \
    --gradient_type "$gradient_type"

python3 -m data_selection.collect_semantic_embedding \
    --train_file "$train_file" \
    --model_path "$base_model" \
    --output_path "$semantic_output_path" \
    --save_interval 5000 \
    --has_input True
