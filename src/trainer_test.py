from trainer.custom_trainer import CustomRLOO
from datasets import load_dataset
from trl.rewards import accuracy_reward
from transformers import AutoTokenizer

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", padding_side='left')
tokenizer.chat_template = "{{ messages[0]['content'] }}"

trainer = CustomRLOO(
    model="EleutherAI/pythia-70m",
    reward_funcs=accuracy_reward,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()