import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DebertaV2Tokenizer
from data.tldr_data_loader import load_tldr_preferences_for_trainer

def reward_collator(data, tokenizer, max_length=512):
    prompts, chosen, rejected = [x["prompt"] for x in data], [x["chosen"] for x in data], [x["rejected"] for x in data]
    inputs_chosen = tokenizer([p + " " + c for p, c in zip(prompts, chosen)], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs_rejected = tokenizer([p + " " + r for p, r in zip(prompts, rejected)], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids_chosen": inputs_chosen["input_ids"], "attention_mask_chosen": inputs_chosen["attention_mask"],
        "input_ids_rejected": inputs_rejected["input_ids"], "attention_mask_rejected": inputs_rejected["attention_mask"],
    }

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        r_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"]).logits
        r_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"]).logits
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
        return (loss, {"r_chosen": r_chosen, "r_rejected": r_rejected}) if return_outputs else loss

def run_rm(args):
    print(f"--- Starting RM (Paper Config: 1 Epoch, Cosine) ---")
    dataset = load_tldr_preferences_for_trainer(pref_split="train")
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.rm_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.rm_model, num_labels=1, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    
    args = TrainingArguments(
        output_dir=args.rm_out,
        learning_rate=args.rm_lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1, # Paper uses 1 epoch
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=1000,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )
    trainer = RewardTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=lambda x: reward_collator(x, tokenizer))
    trainer.train()
    trainer.save_model(args.rm_out)
    tokenizer.save_pretrained(args.rm_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_model", type=str, required=True)
    parser.add_argument("--sft_model", type=str, required=False) # Unused placeholder
    parser.add_argument("--rm_out", type=str, required=True)
    parser.add_argument("--rm_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    run_rm(args)