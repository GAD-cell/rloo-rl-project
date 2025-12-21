import argparse
import torch
import yaml
import sys 
import os 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["WANDB_PROJECT"] = "rloo-commongen"

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RLOOTrainer, RLOOConfig, SFTTrainer, SFTConfig, PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
#from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trainer.custom_trainer import CustomRLOO
from datasets import load_dataset

def get_reward_fn(rm_path, tokenizer, device, max_length=512):
    rm = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, torch_dtype=torch.bfloat16).to(device)
    rm.eval()
    def score(prompts, completions, **kwargs):
        texts = [p + " " + c for p, c in zip(prompts, completions)]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad(): outputs = rm(**inputs)
        return outputs.logits.squeeze(-1).tolist()
    return score, rm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
'''
def prepare_model_with_lora(model, config):
    if config["use_lora"]:
        lora_config = LoraConfig(
        r=config.get('r', 16),
        lora_alpha=config.get('alpha', 32),
        target_modules=config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=config.get('dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model
'''

from trl import AutoModelForCausalLMWithValueHead
from transformers import pipeline

import torch
import torch.nn as nn

class RewardModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.base_model_prefix = getattr(model, 'base_model_prefix', 'model')
        self.name_or_path = getattr(model, 'name_or_path', '')
        
    def score(self, hidden_states):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  
        
        if hasattr(self.model, 'classifier'):
            scores = self.model.classifier(hidden_states)
            if scores.dim() == 3:
                scores = scores[:, 0, :] 
        elif hasattr(self.model, 'score'):
            scores = self.model.score(hidden_states)
            if scores.dim() == 1:
                scores = scores.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        else:
            raise AttributeError("Reward model has neither 'classifier' nor 'score' attribute")
        
        if scores.dim() == 1:
            scores = scores.unsqueeze(-1)
            
        return scores  # Retourne (batch_size, 1)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward tous les autres attributs au modèle wrappé"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

def run_ppo(cfg):
    model_path = cfg["sft_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.get("value_model", model_path),
        num_labels=1,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    base_reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["reward_model"],
        num_labels=1,
        torch_dtype=torch.float32,
        device_map="auto",
        output_hidden_states=True,
    )
    reward_model = RewardModelWrapper(base_reward_model)
    
    def prepare_dataset(split):
        ds = load_dataset("allenai/common_gen", split=split)
        def format_fn(ex):
            prompt = f"Concepts: {', '.join(ex['concepts'])}\nSentence: "
            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=cfg.get("max_prompt_length", 512),
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        return ds.map(format_fn, remove_columns=ds.column_names)
    
    train_dataset = prepare_dataset("train")
    eval_dataset = prepare_dataset("validation")
    
    config_ppo = PPOConfig(
        learning_rate=cfg.get("learning_rate", 1.41e-5),
        per_device_train_batch_size=cfg.get("batch_size", 16),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        num_ppo_epochs=cfg.get("num_ppo_epochs", 4),
        num_mini_batches=cfg.get("num_mini_batches", 1),
        total_episodes=cfg.get("total_episodes", 10000),
        local_rollout_forward_batch_size=cfg.get("local_rollout_forward_batch_size", 16),
        response_length=cfg.get("response_length", 128),
        temperature=cfg.get("temperature", 1.0),
        missing_eos_penalty=cfg.get("missing_eos_penalty", None),
    )
    
    trainer = PPOTrainer(
        args=config_ppo,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Modèle sauvegardé dans {cfg['output_dir']}")

def run_rloo(cfg):
    model_path = cfg["sft_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32, 
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["reward_model"],
        num_labels=1,
        dtype=torch.float32,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg["reward_model"])

    def prepare_dataset(split):
        ds = load_dataset("allenai/common_gen", split=split)
        def format_fn(ex):
            return {"prompt": f"Concepts: {', '.join(ex['concepts'])}\nSentence: "}
        return ds.map(format_fn, remove_columns=ds.column_names)

    train_dataset = prepare_dataset("train")
    eval_dataset = prepare_dataset("validation")



    def custom_reward_function(prompts, completions, **kwargs):
        inputs = reward_tokenizer(
            completions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to(reward_model.device)
        
        with torch.no_grad():
            reward_outputs = reward_model(**inputs)
            rewards = reward_outputs.logits.squeeze(-1)

        empty_penalty = torch.tensor([
        -2.0 if len(c.strip()) == 0 else  
        -1.0 if len(c.strip().split()) < 3 else 
        0.0 
        for c in completions
        ], device=rewards.device) 
        
        final_rewards = rewards + empty_penalty
        return final_rewards


    config_rloo = RLOOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["k"],
        max_grad_norm=1.0,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["lr"]),
        bf16=cfg["bf16"],
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_prompt_length=64,
        max_completion_length=32,
        report_to="wandb",
    )

    trainer = RLOOTrainer(
        model=policy_model,
        reward_funcs=custom_reward_function,
        args=config_rloo,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    
    trainer.save_model(cfg["output_dir"])


def run_custom_rloo(cfg):
    model_path = cfg["sft_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # En RL, le padding à gauche est obligatoire pour la génération
    tokenizer.padding_side = "left"
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32, 
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["reward_model"],
        num_labels=1,
        dtype=torch.float32,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg["reward_model"])
    
    def prepare_dataset(split):
        ds = load_dataset("allenai/common_gen", split=split)
        def format_fn(ex):
            return {"prompt": f"Concepts: {', '.join(ex['concepts'])}\nSentence: "}
        return ds.map(format_fn, remove_columns=ds.column_names)

    train_dataset = prepare_dataset("train")
    eval_dataset = prepare_dataset("validation")



    def custom_reward_function(prompts, completions, **kwargs):
        inputs = reward_tokenizer(
            completions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to(reward_model.device)
        
        with torch.no_grad():
            reward_outputs = reward_model(**inputs)
            rewards = reward_outputs.logits.squeeze(-1)

        empty_penalty = torch.tensor([
        -2.0 if len(c.strip()) == 0 else  
        -1.0 if len(c.strip().split()) < 3 else 
        0.0 
        for c in completions
        ], device=rewards.device) 
        
        final_rewards = rewards + empty_penalty
        return final_rewards


    config_rloo = RLOOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["k"],
        max_grad_norm=1.0,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["lr"]),
        bf16=cfg["bf16"],
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_prompt_length=64,
        max_completion_length=32,
        report_to="wandb",
    )

    trainer = CustomRLOO(
        model=policy_model,
        reward_funcs=custom_reward_function,
        args=config_rloo,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    
    trainer.save_model(cfg["output_dir"])

def run_raft(config):
    print(f"RAFT(k={config.k})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model, torch_dtype=torch.bfloat16).to(device)

    reward_fn, _ = get_reward_fn(config.reward_model, tokenizer, device)
    ds = load_tldr_preferences_for_trainer(pref_split="train")
    eval_ds = load_tldr_preferences_for_trainer(pref_split="validation")
    prompts = ds["prompt"]
    raft_data = {"text": []}
    batch_size = config.batch_size * 2 
    policy.eval()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = policy.generate(**inputs, max_new_tokens=100, num_return_sequences=config.k, do_sample=True, temperature=1.0)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for p_idx, prompt in enumerate(batch_prompts):
            start = p_idx * config.k
            cands = decoded[start : start + config.k]
            scores = reward_fn([prompt]*config.k, [c[len(prompt):] for c in cands])
            raft_data["text"].append(cands[scores.index(max(scores))])
            
    del policy; torch.cuda.empty_cache()
    train_ds = Dataset.from_dict(raft_data)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model, torch_dtype=torch.bfloat16, device_map="auto")
    rm_callback = RewardEvalCallback(config.reward_model, eval_ds, tokenizer, policy.device)

    sft_args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        num_train_epochs=1,
        bf16=config.bf16,
        dataset_text_field="text",
        eval_strategy="steps",
        eval_steps=50,
        report_to="none"
    )
    trainer = SFTTrainer(model=policy, train_dataset=train_ds, eval_dataset=eval_ds, config=sft_args, processing_class=tokenizer, callbacks=[rm_callback])
    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to YAML config file'
    )

    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        login(token=hf_token)

    if config["alg"] == "rloo": run_rloo(config)
    elif config["alg"] == "custom-rloo": run_custom_rloo(config)
    elif config["alg"] == "ppo": run_ppo(config)
    else: run_raft(config)