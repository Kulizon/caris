.PHONY: install requirements run clean test split-dataset fine-tune evaluate download-data

install: requirements download-data

requirements:
	pip install -r requirements.txt

download-data:
	curl -s -L -o test_data.json https://iconclass.org/testset/data.json
	curl -s -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

run:
	python main.py

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

evaluate: split-dataset download-data
	python evaluate.py
