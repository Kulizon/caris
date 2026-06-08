.PHONY: install requirements requirements-gemma run clean test split-dataset fine-tune evaluate download-data vector-base

install: requirements download-data vector-base

requirements:
	pip install -r requirements.txt

requirements-gemma:
	pip install ollama
	pip install chromadb
	ollama pull gemma4:12b
	ollama pull mxbai-embed-large

download-data:
	curl -s -L -o test_data.json https://iconclass.org/testset/data.json
	mkdir -p models
	curl -s -L -o models/yolo26n.pt https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26n.pt
	mkdir -p datasets
	curl -s -L -o datasets/iconclass_dataset.zip https://iconclass.org/testset/779ba2ca9e977c58d818e3823a676973.zip

vector-base:
	python embeddings_utils.py

run:
	python main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

split-dataset:
	rm -rf eval_data/ tune_data/
	python split_dataset.py

evaluate: split-dataset
	python evaluate.py
