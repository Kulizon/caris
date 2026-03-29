.PHONY: install requirements run clean test split-dataset fine-tune evaluate download-data

install: requirements download-data

requirements:
	pip install -r requirements.txt

download-data:
	curl -s -L -o test_data.json https://iconclass.org/testset/data.json
	mkdir -p models
	curl -s -L -o models/yolo26n.pt https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26n.pt

run:
	python main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

split-dataset:
	rm -r eval_data/ tune_data/
	python split_dataset.py; \

evaluate: split-dataset download-data
	python evaluate.py
