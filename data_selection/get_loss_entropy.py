import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import os
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, SequentialSampler
import pandas as pd
import torch.nn.functional as F

os.environ["WANDB_MODE"] = "offline"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="sgd")  # Set to sgd but not actually used
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # Disable learning rate scheduler
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate scheduler type."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        self.targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(self.sources, self.targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class CustomTrainer(Trainer):
    """Trainer that only computes loss and entropy, without updating parameters."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize additional attributes
        self.step_losses = []  # Store loss for each step
        self.step_entropies = []  # Store entropy for each step
    
    def get_train_dataloader(self) -> DataLoader:
        """Use SequentialSampler."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = SequentialSampler(self.train_dataset)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        """Compute loss and entropy without backpropagation."""
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            
            # Extract loss
            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            
            # Compute batch average entropy
            logits = outputs.logits
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(prob * log_prob, dim=-1)  # Entropy for each token
            
            # Create a mask like labels, ignoring IGNORE_INDEX
            non_ignore_mask = inputs["labels"] != IGNORE_INDEX
            valid_entropies = entropy[non_ignore_mask]
            
            # Compute batch average entropy
            if valid_entropies.numel() > 0:
                batch_entropy = valid_entropies.mean().item()
            else:
                batch_entropy = 0.0

        # Save results for callback
        self.outputs = outputs
        self.labels = inputs.get("labels", None)
        self.batch_entropy = batch_entropy
        self.batch_loss = loss.item()
        
        # Store current step's loss and entropy
        self.step_losses.append(self.batch_loss)
        self.step_entropies.append(self.batch_entropy)
        
        # Return loss value (but not used for optimization)
        return loss.detach()

    # Disable all optimization operations
    def backward(self, loss: torch.Tensor, **kwargs):
        """Disable backpropagation."""
        pass
    
    def optimizer_step(self, *args, **kwargs):
        """Disable optimizer step."""
        pass
    
    def lr_scheduler_step(self, *args, **kwargs):
        """Disable learning rate scheduling."""
        pass

class StepMetricsCallback(TrainerCallback):
    """Callback to save loss and entropy for each training step."""
    def __init__(self, trainer: Trainer, output_file: str):
        self.trainer = trainer
        self.output_file = output_file
        self.metrics_data = []  # Store metrics for each step
        self.current_step = 0

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Reset metrics at the start of a step."""
        self.current_step = state.global_step

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save loss and entropy at the end of a step."""
        # Ensure trainer has current step data
        if hasattr(self.trainer, 'batch_loss') and hasattr(self.trainer, 'batch_entropy'):
            step_metrics = {
                "step": self.current_step,
                "loss": self.trainer.batch_loss,
                "entropy": self.trainer.batch_entropy
            }
            
            # Add current step's metrics
            self.metrics_data.append(step_metrics)
            
            # Save periodically
            if self.current_step % 100 == 0:
                self.save_metrics()
        else:
            print(f"Step {self.current_step}: No metrics available")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save all metrics at the end of training."""
        self.save_metrics()
        print("\nTraining step metrics saved.")

    def save_metrics(self):
        """Save metrics to a CSV file."""
        if self.metrics_data:
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.output_file, index=False)
            print(f"Step metrics saved to {self.output_file}")
        else:
            print("No metrics data to save.")

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        data_collator=data_module["data_collator"],
        optimizers=(None, None)  # Disable optimizer
    )

    # Add step metrics callback
    trainer.add_callback(StepMetricsCallback(
        trainer=trainer,
        output_file="step_metrics_epoch1.csv"
    ))
    
    trainer.lr_scheduler = None  # Remove scheduler

    trainer.train()

if __name__ == "__main__":
    train()