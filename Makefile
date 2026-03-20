.PHONY: help install requirements run clean test split-dataset fine-tune evaluate download-data

help:
	@echo "CARIS Project"
	@echo ""
	@echo "Available targets:"
	@echo "  make install        - Install all dependencies and download data"
	@echo "  make requirements   - Install from requirements.txt"
	@echo "  make download-data  - Download required data file"
	@echo "  make run            - Run the main project"
	@echo "  make clean          - Remove Python cache files"
	@echo "  make test           - Run the test file"
	@echo "  make split-dataset  - Split Iconclass dataset into tune/eval"
	@echo "  make fine-tune      - Fine-tune YOLO on Iconclass dataset"
	@echo "  make evaluate       - Evaluate CARIS on the eval dataset"
	@echo ""

install: requirements download-data

requirements:
	pip install -r requirements.txt

download-data:
	curl -s -L -o test_data.json https://iconclass.org/testset/data.json
	curl -s -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

run:
	python main.py

test:
	python test.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

split-dataset:
	@if [ ! -f eval_data/data.json ] || [ ! -f tune_data/data.json ]; then \
		python split_dataset.py; \
	else \
		echo "Dataset already split (eval_data/ and tune_data/ exist). Skipping."; \
	fi

fine-tune: split-dataset download-data
	python fine_tune.py

evaluate: split-dataset download-data
	python evaluate.py
