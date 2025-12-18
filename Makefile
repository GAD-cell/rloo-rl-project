PYTHON_VERSION := 3.11
VENV := .venv
PYTHON := ./$(VENV)/bin/python3

RM_DATA_SCRIPT := src/data/rm_dataset.py
RM_TRAIN_SCRIPT := src/train/rm.py
CONFIG_FILE_RM := src/config/rm_config.yaml

SFT_TRAIN_SCRIPT := src/train/sft.py
CONFIG_FILE_SFT := src/config/sft_config.yaml

RL_TRAIN_SCRIPT := src/train/rl.py
CONFIG_FILE_RL := src/config/rl_config.yaml
.PHONY: env rm_data train_rm

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(VENV) --python $(PYTHON_VERSION) --no-project
	@$(VENV)/bin/python -m pip install -r requirements.txt
	@echo "Evaluation environment ready."

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

train_rl:
	@echo "Starting RL Training with config $(CONFIG_FILE_RL)..."
	@$(PYTHON) $(RL_TRAIN_SCRIPT) --config $(CONFIG_FILE_RL)
	@echo "Training finished. Model saved in ./models/rl_model"