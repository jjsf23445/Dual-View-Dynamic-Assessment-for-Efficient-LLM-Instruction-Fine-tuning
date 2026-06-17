import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset


def get_training_dataset(train_files: List[str], tokenizer, max_seq_length):
    """ get training dataset """
    raw_datasets = load_raw_dataset(train_files)
    lm_datasets = encode_data(raw_datasets, tokenizer, max_seq_length)
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str]):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    return processed_datasets


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_instruction_format"):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_instruction_format"):
    """ get encode function based on the dataset. """
    if "instruction" in raw_datasets.column_names and "input" in raw_datasets.column_names and "output" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_instruction_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have 'instruction', 'input', and 'output' in your column names.")
    return encode_function


def encode_with_instruction_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'instruction', 'input', and 'output' fields.
    We concatenate instruction, input, and output and tokenize them together.
    '''
    # Concatenate instruction, input, and output
    example_text = example['instruction'] + ' ' + example['input'] + ' ' + example['output']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_instruction_input = tokenizer(
        example['instruction'] + ' ' + example['input'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # Mask the instruction and input part for avoiding loss
    labels[:, :tokenized_instruction_input.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }