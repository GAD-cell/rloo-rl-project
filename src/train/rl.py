import argparse
import torch
import yaml
import sys 
import os 
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["WANDB_PROJECT"] = "rloo-commongen"

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RLOOTrainer, RLOOConfig, SFTTrainer, SFTConfig, AutoModelForCausalLMWithValueHead
from trl.experimental.ppo import  PPOConfig, PPOTrainer
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
    
def run_ppo(cfg):
    model_path = cfg["sft_model"]
    reward_model_path = cfg["reward_model"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": "auto",
    }

    policy = AutoModelForCausalLM.from_pretrained(
        model_path, 
        **model_kwargs
    )
    from transformers import GenerationConfig

    '''
    gen_cfg = GenerationConfig(
        min_new_tokens=10,
        max_new_tokens=16,
        temperature = 1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    policy.generation_config = gen_cfg
    '''
    value_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        **model_kwargs
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        **model_kwargs,
    )
    original_rm_forward = reward_model.forward
    def safe_forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 0:
            device = input_ids.device
            input_ids = torch.full((input_ids.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            if attention_mask is not None:
                attention_mask = torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=device)
        
        return original_rm_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    original_value_forward = value_model.forward
    def safe_forward_value(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 0:
            device = input_ids.device
            input_ids = torch.full((input_ids.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            if attention_mask is not None:
                attention_mask = torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=device)
    
        return original_value_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    from types import MethodType
    reward_model.forward = MethodType(safe_forward, reward_model)
    value_model.forward = MethodType(safe_forward_value, value_model)

    def prepare_dataset(split):
        ds = load_dataset("allenai/common_gen", split=split)
        def tokenize_fn(ex):
            prompt_text = f"Concepts: {', '.join(ex['concepts'])}\nSentence: "
            tokenized = tokenizer(
                prompt_text, 
                truncation=True, 
                max_length=64,
                padding=False 
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }
        return ds.map(
            tokenize_fn, 
            remove_columns=ds.column_names,
            batched=False
        )

    train_dataset = prepare_dataset("train")
    validation_dataset = prepare_dataset("validation")

    config_ppo = PPOConfig(
        learning_rate=cfg.get("lr", 1.41e-5),
        temperature=1.0,
        response_length=16,
        batch_size=cfg.get("batch_size", 16),
        mini_batch_size=cfg.get("mini_batch_size", 16),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        eval_steps=1000
    )
    trainer = PPOTrainer(
        args=config_ppo,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )
    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

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
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    def prepare_dataset(split):
        ds = load_dataset("allenai/common_gen", split=split)
        def format_fn(ex):
            return {"prompt": f"Concepts: {', '.join(ex['concepts'])}\nSentence: "}
        return ds.map(format_fn, remove_columns=ds.column_names)

    train_dataset = prepare_dataset("train")
    eval_dataset = prepare_dataset("validation")



    def custom_reward_function(prompts, completions, **kwargs):
        cleaned_completions = [c if len(c.strip()) > 0 else " " for c in completions]
        
        inputs = reward_tokenizer(
            cleaned_completions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to(reward_model.device)
        
        with torch.no_grad():
            reward_outputs = reward_model(**inputs)
            rewards = reward_outputs.logits.squeeze(-1)

        penalties = []
        for c in completions:
            stripped = c.strip()
            if len(stripped) == 0:
                penalties.append(-2.0)
            elif len(stripped.split()) < 3:
                penalties.append(-1.0)
            else:
                penalties.append(0.0)
        
        #penalty_tensor = torch.tensor(penalties, device=rewards.device)
        return rewards #+ penalty_tensor


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
        eval_steps=500,
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
        
        final_rewards = rewards #+ empty_penalty
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