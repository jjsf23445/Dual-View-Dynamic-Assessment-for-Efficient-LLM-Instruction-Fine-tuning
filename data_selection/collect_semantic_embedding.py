import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import numpy as np
import os
import argparse

def embed_texts_batched(texts, batch_size=30, save_interval=5000, save_path=None):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}  # Ensure data is moved to GPU
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        all_embeddings.extend(embeddings)

        # Save embeddings every save_interval samples
        if (i + batch_size) % save_interval == 0 and save_path is not None:
            np.save(f'{save_path}/embeddings_{i // save_interval}.npy', np.array(all_embeddings))
            all_embeddings = []  # Clear list for the next save

    # Save remaining embeddings
    if all_embeddings:
        np.save(f'{save_path}/embeddings_final.npy', np.array(all_embeddings))

def preprocess_dataset(raw_data, has_input=True):
    data = []
    for e in raw_data:
        if has_input:
            data.append({"instruction": e["instruction"].strip(), "input": e["input"].strip(), "output": e["output"].strip()})
        else:
            data.append({"instruction": e["instruction"].strip(), "output": e["output"].strip()})
    return data

def generate_prompt(instruction, input=None, output=None):
    if input is not None and input != "":
        with_query = f"Instruction:\n{instruction}\nInput:\n{input}\nResponse:\n"
    else:
        with_query = f"Instruction:\n{instruction}\nResponse:\n"

    if output is not None:
        with_query_and_answer = f"{with_query}{output}"
    else:
        with_query_and_answer = with_query

    return with_query, with_query_and_answer

def load_sample(file_name, has_input=True):
    with open(file_name, "r") as f:
        data = json.load(f)
        data = preprocess_dataset(data, has_input=has_input)
        print(f"Data loaded: {file_name}.")
    
    example_list = [[e["instruction"], e.get("input", ""), e["output"]] for e in data]
    prompted_examples = []
    for instruction, input, output in example_list:
        _, query_with_answer = generate_prompt(instruction, input, output)  # Use the prompt with output
        prompted_examples.append(query_with_answer)  # Use the prompt with output
    return prompted_examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed texts and save embeddings")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file (JSON format)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save embeddings")
    parser.add_argument("--has_input", type=bool, default=False, help="Whether the dataset contains an input field")
    parser.add_argument("--save_interval", type=int, default=5000, help="Interval at which embeddings are saved")
    args = parser.parse_args()

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)  # Ensure the model is moved to GPU
    model.eval()

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # Ensure the save path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Load sample
    sample = load_sample(args.train_file, has_input=args.has_input)
    print("START EMBEDDING ..." * 3)
    embed_texts_batched(sample, batch_size=30, save_interval=args.save_interval, save_path=args.output_path)
    print("Embedding process completed.")
