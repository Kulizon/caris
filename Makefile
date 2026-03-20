.PHONY: help install requirements run clean test

help:
	@echo "CARIS Project"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install all dependencies and download data"
	@echo "  make requirements  - Install from requirements.txt"
	@echo "  make download-data - Download required data file"
	@echo "  make run           - Run the main project"
	@echo "  make clean         - Remove Python cache files"
	@echo "  make test          - Run the test file"
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
