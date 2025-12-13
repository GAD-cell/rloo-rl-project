import torch
import os
import json
import numpy as np
from transformers import TrainerCallback, AutoModelForSequenceClassification, AutoTokenizer

class RewardEvalCallback(TrainerCallback):
    """
    J'ai plus accès à wandb
    """
    def __init__(self, rm_path, eval_dataset, tokenizer, device, num_samples=64):
        self.rm_path = rm_path
        self.eval_dataset = eval_dataset.select(range(num_samples)) 
        self.tokenizer = tokenizer
        self.device = device
        self.rm = None
        self.rm_tokenizer = None
        self.log_file = None

    def _load_rm(self):
        if self.rm is None:
            self.rm = AutoModelForSequenceClassification.from_pretrained(
                self.rm_path, num_labels=1, torch_dtype=torch.bfloat16
            ).to(self.device)
            self.rm.eval()
            self.rm_tokenizer = AutoTokenizer.from_pretrained(self.rm_path)

    def on_evaluate(self, args, state, control, model, **kwargs):
        self._load_rm()
        if self.log_file is None:
            self.log_file = os.path.join(args.output_dir, "eval_rewards.jsonl")
            os.makedirs(args.output_dir, exist_ok=True)
        prompts = self.eval_dataset["prompt"]
        model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7 
            )
        full_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rm_inputs = self.rm_tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            rm_outputs = self.rm(**rm_inputs)
            scores = rm_outputs.logits.squeeze(-1).float().cpu().numpy()
            
        mean_reward = float(np.mean(scores))
        
        log_entry = {"step": state.global_step, "reward": mean_reward}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        print(f"\n[Eval] Step {state.global_step}: Mean Reward = {mean_reward:.4f} (Saved to {self.log_file})")