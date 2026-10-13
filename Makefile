.PHONY: help install test unit-test audit reproduce artifacts dashboard clean

help:
	@echo "Available commands:"
	@echo "  make install      Install package and dependencies in editable mode"
	@echo "  make test         Run all unit and integration tests"
	@echo "  make unit-test    Run 102 fast unit tests"
	@echo "  make audit        Run clinical safety and data leakage audits"
	@echo "  make reproduce    Run the complete V10 reproducible master pipeline"
	@echo "  make artifacts    Generate camera-ready figures (PDF) and LaTeX tables"
	@echo "  make dashboard    Launch the Streamlit clinical decision support app"
	@echo "  make clean        Clean build and cache files"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/unit/ tests/integration/ -v

unit-test:
	python -m pytest tests/unit/ -v

audit:
	python -m pytest tests/audits/ -v

reproduce:
	python experiments/pipelines/run_v10_master_pipeline.py

artifacts:
	python scripts/generation/generate_paper_artifacts_v5.py

dashboard:
	streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
