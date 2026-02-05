# Horde Agent Makefile

.PHONY: help install install-cpu install-gpu ui workflow clean test

help:
	@echo "Horde Agent - LLM Training Workflow"
	@echo ""
	@echo "Available commands:"
	@echo "  make install-cpu  - Install CPU dependencies (local dev)"
	@echo "  make install-gpu  - Install GPU dependencies (production)"
	@echo "  make ui           - Run Streamlit UI"
	@echo "  make workflow     - Run workflow from CLI"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean generated files"

install-cpu:
	pip install -r requirements-cpu.txt

install-gpu:
	pip install -r requirements.txt

install: install-cpu

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
