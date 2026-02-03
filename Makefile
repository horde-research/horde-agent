# Horde Agent Makefile

.PHONY: help install ui workflow clean test

help:
	@echo "Horde Agent - LLM Training Workflow"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make ui         - Run Streamlit UI"
	@echo "  make workflow   - Run workflow from CLI"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"

install:
	pip install -r requirements.txt

ui:
	streamlit run ui/app.py

workflow:
	python scripts/run_workflow.py --input "$(INPUT)"

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/logs data/state data/reports
