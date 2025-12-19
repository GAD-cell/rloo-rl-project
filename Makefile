PYTHON_VERSION := 3.11
VENV := .venv
PYTHON := ./$(VENV)/bin/python3

RM_DATA_SCRIPT := src/data/rm_dataset.py
RM_TRAIN_SCRIPT := src/train/rm.py
SFT_TRAIN_SCRIPT := src/train/sft.py
RL_TRAIN_SCRIPT := src/train/rl.py
SFT_GSM8K_TRAIN_SCRIPT := src/train/sft_gsm8k.py
RL_GSM8K_TRAIN_SCRIPT := src/train/rl_gsm8k.py
CONFIG_FILE_RM := src/config/rm_config.yaml
CONFIG_FILE_SFT := src/config/sft_config.yaml
CONFIG_FILE_RL := src/config/rl_config.yaml
CONFIG_FILE_SFT_GSM8K := src/config/sft_gsm8k_config.yaml
CONFIG_FILE_RL_GSM8K := src/config/rl_gsm8k_config.yaml

.PHONY: env rm_data train_rm train_sft train_sft_gsm8k train_rl train_rl_gsm8k clean

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up environment..."
	@rm -rf $(VENV)
	@uv venv $(VENV) --python $(PYTHON_VERSION)
	@echo "Installing dependencies..."
	@uv pip install -r requirements.txt --python $(PYTHON)
	@echo "Environment ready. Activate with: source $(VENV)/bin/activate"

rm_data:
	@$(PYTHON) $(RM_DATA_SCRIPT)

train_rm:
	@echo "Starting RM Training with config $(CONFIG_FILE_RM)..."
	@$(PYTHON) $(RM_TRAIN_SCRIPT) --config $(CONFIG_FILE_RM)
	@echo "Training finished. Model saved in ./models/rm_model"

train_sft:
	@echo "Starting SFT Training with config $(CONFIG_FILE_SFT)..."
	@$(PYTHON) $(SFT_TRAIN_SCRIPT) --config $(CONFIG_FILE_SFT)
	@echo "Training finished. Model saved in ./models/sft_model"

train_sft_gsm8k:
	@echo "Starting GSM8K SFT Training with config $(CONFIG_FILE_SFT_GSM8K)..."
	@$(PYTHON) $(SFT_GSM8K_TRAIN_SCRIPT) --config $(CONFIG_FILE_SFT_GSM8K)
	@echo "Training finished. Model saved to output_dir set in $(CONFIG_FILE_SFT_GSM8K)"

train_rl:
	@echo "Starting RL Training with config $(CONFIG_FILE_RL)..."
	@$(PYTHON) $(RL_TRAIN_SCRIPT) --config $(CONFIG_FILE_RL)
	@echo "Training finished. Model saved in ./models/rl_model"

train_rl_gsm8k:
	@echo "Starting GSM8K RL Training with config $(CONFIG_FILE_RL_GSM8K)..."
	@$(PYTHON) $(RL_GSM8K_TRAIN_SCRIPT) --config $(CONFIG_FILE_RL_GSM8K)
	@echo "Training finished. Model saved to output_dir set in $(CONFIG_FILE_RL_GSM8K)"

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@rm -rf __pycache__ src/__pycache__
	@echo "Cleanup complete"
