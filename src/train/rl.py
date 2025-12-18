import argparse
import torch
import yaml
import sys 
import os 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RLOOTrainer, RLOOConfig, SFTTrainer, SFTConfig
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



def run_rloo(cfg):
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

    def prepare_dataset(split):
        ds = load_dataset("allenai/common_gen", split=split)
        def format_fn(ex):
            return {"prompt": f"Concepts: {', '.join(ex['concepts'])}\nSentence: "}
        return ds.map(format_fn, remove_columns=ds.column_names)

    train_dataset = prepare_dataset("train")
    eval_dataset = prepare_dataset("validation")



    def custom_reward_function(prompts, completions, **kwargs):
        texts = [p + c for p, c in zip(prompts, completions)]
        
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(reward_model.device)
        
        with torch.no_grad():
            reward_outputs = reward_model(**inputs)
            rewards = reward_outputs.logits.squeeze(-1) 

        final_rewards = rewards 
        
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


def run_custom_rloo(config):
    print(f"RLOO custom(k={config['k']})")
    dataset = load_tldr_preferences_for_trainer(pref_split="train")
    eval_ds = load_tldr_preferences_for_trainer(pref_split="validation")
    print(config["sft_model"])
    tokenizer = AutoTokenizer.from_pretrained(config["sft_model"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    ref_policy = AutoModelForCausalLM.from_pretrained(config["sft_model"], torch_dtype=torch.bfloat16, device_map="auto")

    reward_fn, _ = get_reward_fn(config["reward_model"], tokenizer, ref_policy.device)
    rm_callback = RewardEvalCallback(config["reward_model"], eval_ds, tokenizer, ref_policy.device)

    config_rloo = RLOOConfig(
        output_dir=config["output_dir"],
        num_generations=config["k"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=config["bf16"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_prompt_length=256,
        report_to="none"
    )
    trainer = CustomRLOO(
            model=ref_policy,                 
            reward_funcs=reward_fn,        
            config=config_rloo,                   
            train_dataset=dataset,
            eval_dataset=eval_ds,
            processing_class=tokenizer,   
            callbacks=[rm_callback]
        )    
    trainer.train()
    trainer.save_model(config.output_dir)

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
    else: run_raft(config)