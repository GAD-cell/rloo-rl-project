import random
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
vocab_list = [w.replace('Ġ', '') for w in tokenizer.get_vocab().keys() if w.startswith('Ġ') and len(w) > 3]

dataset = load_dataset("allenai/common_gen", split="train")
all_concepts = list(set([c for sublist in dataset['concepts'] for c in sublist]))

def get_distractor_concept(current_concepts):
    distractor = random.choice(all_concepts)
    while distractor in current_concepts:
        distractor = random.choice(all_concepts)
    return distractor

def complex_negative_generator(example, p_shuffle=0.3, p_drop=0.3, p_insert_vocab=0.4, p_insert_distractor=0.2):
    concepts = example['concepts']
    prompt = ", ".join(concepts)
    positive = example['target']
    words = positive.split()
    
    if random.random() < p_drop and len(words) > 3:
        target_drop = random.choice(concepts)
        words = [w for w in words if target_drop.lower() not in w.lower()]
    
    if random.random() < p_insert_vocab:
        words.insert(random.randint(0, len(words)), random.choice(vocab_list))
        
    if random.random() < p_insert_distractor:
        words.insert(random.randint(0, len(words)), get_distractor_concept(concepts))
    
    if random.random() < p_shuffle:
        random.shuffle(words)
        
    negative = " ".join(words)
    
    if negative == positive:
        random.shuffle(words)
        negative = " ".join(words)

    return {
        "prompt": prompt,
        "chosen": positive,
        "rejected": negative
    }

small_ds = dataset.select(range(2000)).map(
    lambda x: complex_negative_generator(x), 
    remove_columns=dataset.column_names
)

small_ds.to_json("src/data/commongen_rm_dataset.jsonl")

