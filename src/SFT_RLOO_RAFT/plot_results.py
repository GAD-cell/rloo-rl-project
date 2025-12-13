import json
import matplotlib.pyplot as plt
import os

def load_logs(filepath):
    steps, rewards = [], []
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return [], []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            steps.append(data["step"])
            rewards.append(data["reward"])
    return steps, rewards

def plot(paths, labels):
    plt.figure(figsize=(10, 6))
    for path, label in zip(paths, labels):
        steps, rewards = load_logs(path)
        if steps:
            plt.plot(steps, rewards, marker='o', label=label)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Reward Model Score")
    plt.title("Reward vs Steps (Reproduction)")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_comparison.png")
    print("Graph saved to reward_comparison.png")

if __name__ == "__main__":
    paths = [
        "checkpoints/dpo/eval_rewards.jsonl",
        "checkpoints/rloo/eval_rewards.jsonl",
        "checkpoints/raft/eval_rewards.jsonl"
    ]
    labels = ["DPO", "RLOO (k=2)", "RAFT (k=2)"]
    plot(paths, labels)