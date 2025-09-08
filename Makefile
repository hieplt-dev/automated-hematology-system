PYTHON := python
TEST_PATH := tests/
SRC_PATH := src/

.PHONY: train infer
train:  ## Run train
	$(PYTHON) -m src.ahs.cli.train_cli

infer:
	$(PYTHON) -m src.ahs.cli.infer_cli --img_path $(img_path)