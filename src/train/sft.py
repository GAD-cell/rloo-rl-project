import torch 
import argparse
import yaml

from trl import SFTTrainer, SFTConfig

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def train_sft(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset(cfg["dataset"]["name"], split="train", trust_remote_code=True)

    def format_fn(ex):
        return {
            "text": f"Concepts: {', '.join(ex['concepts'])}\nSentence: {ex['target']}"
        }
    
    
    train_dataset = raw_dataset.map(format_fn, remove_columns=raw_dataset.column_names)
    print(train_dataset[0])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], 
        torch_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
        device_map="auto"
    )

    sft_config = SFTConfig(
        output_dir=cfg["model"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=1, 
        warmup_steps=cfg["training"]["warmup_steps"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["lr"]),
        logging_steps=cfg["training"]["logging_steps"],
        optim="adamw_torch",
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type="linear",
        seed=cfg["training"]["seed"],
        report_to="wandb", 
        project="sft_rloo"

    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=sft_config 
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
        train_sft(cfg)