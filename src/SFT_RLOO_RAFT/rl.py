import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RLOOTrainer, RLOOConfig, SFTTrainer, SFTConfig
from datasets import Dataset
from data.tldr_data_loader import load_tldr_preferences_for_trainer
from common_utils import RewardEvalCallback

def get_reward_fn(rm_path, tokenizer, device, max_length=512):
    rm = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, torch_dtype=torch.bfloat16).to(device)
    rm.eval()
    def score(prompts, completions, **kwargs):
        texts = [p + " " + c for p, c in zip(prompts, completions)]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad(): outputs = rm(**inputs)
        return outputs.logits.squeeze(-1).tolist()
    return score, rm

def run_rloo(args):
    print(f"RLOO(k={args.k})")
    dataset = load_tldr_preferences_for_trainer(pref_split="train")
    eval_ds = load_tldr_preferences_for_trainer(pref_split="validation")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.bfloat16, device_map="auto")
    reward_fn, _ = get_reward_fn(args.reward_model, tokenizer, policy.device)
    rm_callback = RewardEvalCallback(args.reward_model, eval_ds, tokenizer, policy.device)

    config = RLOOConfig(
        output_dir=args.output_dir,
        num_generations=args.k,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_prompt_length=256,
        report_to="none"
    )
    trainer = RLOOTrainer(
            model=ref_policy,                 
            reward_funcs=reward_fn,        
            args=config,                   
            train_dataset=dataset,
            eval_dataset=eval_ds,
            processing_class=tokenizer,   
            callbacks=[rm_callback]
        )    
    trainer.train()
    trainer.save_model(args.output_dir)

def run_raft(args):
    print(f"RAFT(k={args.k})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.bfloat16).to(device)
    reward_fn, _ = get_reward_fn(args.reward_model, tokenizer, device)
    ds = load_tldr_preferences_for_trainer(pref_split="train")
    eval_ds = load_tldr_preferences_for_trainer(pref_split="validation")
    prompts = ds["prompt"]
    raft_data = {"text": []}
    batch_size = args.batch_size * 2 
    policy.eval()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = policy.generate(**inputs, max_new_tokens=100, num_return_sequences=args.k, do_sample=True, temperature=1.0)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for p_idx, prompt in enumerate(batch_prompts):
            start = p_idx * args.k
            cands = decoded[start : start + args.k]
            scores = reward_fn([prompt]*args.k, [c[len(prompt):] for c in cands])
            raft_data["text"].append(cands[scores.index(max(scores))])
            
    del policy; torch.cuda.empty_cache()
    train_ds = Dataset.from_dict(raft_data)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.bfloat16, device_map="auto")
    rm_callback = RewardEvalCallback(args.reward_model, eval_ds, tokenizer, policy.device)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        num_train_epochs=1,
        bf16=args.bf16,
        dataset_text_field="text",
        eval_strategy="steps",
        eval_steps=50,
        report_to="none"
    )
    trainer = SFTTrainer(model=policy, train_dataset=train_ds, eval_dataset=eval_ds, args=sft_args, processing_class=tokenizer, callbacks=[rm_callback])
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, required=True)
    parser.add_argument("--sft_model", type=str, required=True)
    parser.add_argument("--reward_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.alg == "rloo": run_rloo(args)
    else: run_raft(args)