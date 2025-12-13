import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from data.tldr_data_loader import load_tldr_sft_for_trainer

def run_sft(args):
    print(f"SFT")
    dataset = load_tldr_sft_for_trainer(sft_split="train")
    if "response" in dataset.column_names and "completion" not in dataset.column_names:
        dataset = dataset.rename_column("response", "completion")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def add_eos(ex):
        if not ex["completion"].endswith(tokenizer.eos_token):
            ex["completion"] = ex["completion"] + tokenizer.eos_token
        return ex
    dataset = dataset.map(add_eos)
    attn_implementation = "flash_attention_2" if args.flash_attn else None
    print(f"Using attention implementation: {attn_implementation if attn_implementation else 'default'}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=attn_implementation,
        device_map="auto"
    )
    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=1000,
        save_strategy="epoch",
        bf16=args.bf16,
        report_to="none",
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)  # Paper uses 2 for Pythia
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention 2")
    args = parser.parse_args()
    run_sft(args)
