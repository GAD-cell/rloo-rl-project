import torch
import yaml
import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig


def train_rm(cfg):

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model']['name'], 
        num_labels=1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    original_forward = model.forward
    def patched_forward(*args, **kwargs):
        kwargs.pop("use_cache", None) 
        return original_forward(*args, **kwargs)
    
    model.forward = patched_forward
    
    dataset = load_dataset("json", data_files=cfg['data']['path'], split="train")
    dataset = dataset.train_test_split(test_size=cfg['data']['test_size'])

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(prompt, chosen, truncation=True, max_length=cfg['data']['max_length'])
            tokenized_rejected = tokenizer(prompt, rejected, truncation=True, max_length=cfg['data']['max_length'])

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    tokenized_ds = dataset.map(preprocess_function, batched=True)

    training_args = RewardConfig(
        output_dir=cfg['model']['output_dir'],
        per_device_train_batch_size=cfg['training']['batch_size'],
        num_train_epochs=cfg['training']['epochs'],
        learning_rate=float(cfg['training']['lr']),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg['training']['eval_steps'],
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        center_rewards_coefficient=0.1
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
    )

    trainer.train()
    model.save_pretrained(cfg['model']['output_dir'])
    tokenizer.save_pretrained(cfg['model']['output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        train_rm(cfg)