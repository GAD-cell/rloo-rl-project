# sft_gsm8k.py
import torch
import argparse
import yaml
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

def extract_answer_text(answer_field: str) -> str:
    m = ANSWER_RE.search(answer_field)
    raw = m.group(1).strip() if m else answer_field
    n = NUM_RE.search(raw)
    if not n:
        return raw.strip()
    return n.group(0).replace(",", "")

def make_prompt(q: str) -> str:
    return f"Question: {q.strip()}\nAnswer (just the number):"

def train_sft(cfg):
    def resolve_dtype():
        if cfg["training"].get("bf16", False) and torch.cuda.is_available():
            return torch.bfloat16
        if cfg["training"].get("fp16", False) and torch.cuda.is_available():
            return torch.float16
        print("running on CPU")
        return torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = load_dataset("gsm8k", "main", split="train")

    def format_fn(ex):
        ans = extract_answer_text(ex["answer"])
        prompt = make_prompt(ex["question"])
        return {"text": f"{prompt} {ans}"}

    train_dataset = raw.map(format_fn, remove_columns=raw.column_names)
    print(train_dataset[0])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=resolve_dtype(),
        device_map="auto",
    )

    sft_config = SFTConfig(
        output_dir=cfg["model"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=1,
        warmup_steps=cfg["training"]["warmup_steps"],
        max_grad_norm=1,
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["lr"]),
        logging_steps=cfg["training"]["logging_steps"],
        optim="adamw_torch",
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type="linear",
        seed=cfg["training"]["seed"],
        report_to="wandb",
        project="sft_gsm8k_answer_only",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )
    trainer.train()

    model.save_pretrained(cfg["model"]["output_dir"])
    tokenizer.save_pretrained(cfg["model"]["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_sft(cfg)
