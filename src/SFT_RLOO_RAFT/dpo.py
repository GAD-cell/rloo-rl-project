import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from data.tldr_data_loader import load_tldr_preferences_for_trainer
from common_utils import RewardEvalCallback

def run_dpo(args):
    print("DPO")
    ds = load_tldr_preferences_for_trainer(pref_split="train")
    eval_ds = load_tldr_preferences_for_trainer(pref_split="validation")
    model = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    rm_callback = RewardEvalCallback(args.reward_model_path, eval_ds, tokenizer, model.device)

    cfg = DPOConfig(
        output_dir=args.dpo_out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.dpo_lr,
        lr_scheduler_type="constant_with_warmup", 
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        max_length=args.max_seq_len,
        max_prompt_length=256,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="steps", 
        eval_steps=50,         
        beta=0.1,
        report_to="none" 
    )

    trainer = DPOTrainer(model=model, ref_model=ref_model, args=cfg, train_dataset=ds, eval_dataset=eval_ds, processing_class=tokenizer, callbacks=[rm_callback])
    trainer.train()
    trainer.save_model(args.dpo_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--dpo_out", type=str, required=True)
    parser.add_argument("--dpo_lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    run_dpo(args)