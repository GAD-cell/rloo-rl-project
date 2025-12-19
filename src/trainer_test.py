from trainer.custom_trainer import CustomRLOO
from datasets import load_dataset
from trl.rewards import accuracy_reward
from transformers import AutoTokenizer
from trl import RLOOConfig

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
print(dataset)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", padding_side='left')
tokenizer.chat_template = "{{ messages[0]['content'] }}"

rloo_config = RLOOConfig(
    report_to=["wandb"],            
    project="back_to_basics_rloo",   

    per_device_train_batch_size=8,   
    gradient_accumulation_steps=1,  
    max_steps=400,                   
    learning_rate=1e-6,              
    
    logging_strategy="steps",        
    logging_steps=10,                
    log_completions=True,           
    wandb_log_unique_prompts=True,  
    
    num_generations=4,             
    beta=0.1,                        
    gradient_checkpointing=True,      
)


trainer = CustomRLOO(
    model="EleutherAI/pythia-70m",
    reward_funcs=accuracy_reward,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=rloo_config
)
trainer.train()