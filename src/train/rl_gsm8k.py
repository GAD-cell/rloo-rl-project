# rl_gsm8k.py
import argparse
import torch
import yaml
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RLOOTrainer, RLOOConfig
from datasets import load_dataset

ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

def extract_gold(answer_field: str) -> str:
    m = ANSWER_RE.search(answer_field)
    raw = m.group(1).strip() if m else answer_field
    n = NUM_RE.search(raw)
    if not n:
        return raw.strip()
    return n.group(0).replace(",", "")

def extract_pred(completion: str) -> str:
    # take first number anywhere in completion
    n = NUM_RE.search(completion)
    if not n:
        return ""
    return n.group(0).replace(",", "")

def make_prompt(q: str) -> str:
    return f"Question: {q.strip()}\nAnswer (just the number):"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_rloo(cfg):
    model_path = cfg["sft_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # RL generation: left padding is standard

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if cfg.get("fp16", True) else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    raw_train = load_dataset("gsm8k", "main", split="train")
    raw_eval  = load_dataset("gsm8k", "main", split="test")

    def format_rl(ex):
        return {
            "prompt": make_prompt(ex["question"]),
            "gold": extract_gold(ex["answer"]),
        }

    train_dataset = raw_train.map(format_rl, remove_columns=raw_train.column_names)
    eval_dataset  = raw_eval.map(format_rl, remove_columns=raw_eval.column_names)

    # Rule-based reward: exact match on number string
    def gsm8k_reward_function(prompts, completions, **kwargs):
        # TRL passes a dict of extra columns in kwargs for many trainers;
        # for RLOO it will include the batch from dataset.
        golds = kwargs.get("gold", None)
        if golds is None:
            # fallback: no gold provided (shouldn't happen if dataset has it)
            return torch.zeros(len(completions), device=policy_model.device)

        rewards = []
        for comp, gold in zip(completions, golds):
            pred = extract_pred(comp)
            r = 1.0 if (pred != "" and pred == gold) else 0.0

            # Optional: penalize verbose outputs (keeps "few tokens" behavior)
            # Example penalty: if more than ~6 tokens of output, subtract 0.2
            if len(comp.strip().split()) > 3:
                r -= 0.2
            if pred == "":
                r -= 0.2

            rewards.append(r)

        return torch.tensor(rewards, device=policy_model.device)

    config_rloo = RLOOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["k"],
        max_grad_norm=1.0,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["lr"]),
        bf16=cfg["bf16"],
        fp16=not cfg["bf16"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_prompt_length=128,
        max_completion_length=8,  
        report_to="wandb",
    )

    trainer = RLOOTrainer(
        model=policy_model,
        reward_funcs=gsm8k_reward_function,
        args=config_rloo,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_rloo(cfg)
