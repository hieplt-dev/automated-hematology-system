PYTHON := python
TEST_PATH := tests/
SRC_PATH := src/

.PHONY: train infer
train:  ## Run train
	$(PYTHON) -m src.ahs.cli.train_cli

## Run inference
infer:
	$(PYTHON) -m src.ahs.cli.infer_cli --img_path $(img_path)

## api
api:
	uvicorn src.ahs.api.fast_api:app --host 0.0.0.0 --port 8000 --reload

## Run dynamic quantization
quantize_dynamic:
	$(PYTHON) -m scripts.quantize_dynamic