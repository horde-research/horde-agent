# Horde Agent

LLM Training Agent for SFT/GRPO/DPO workflows.

## Structure

- `agent/` - Core orchestration logic
- `tools/` - Individual tool implementations
- `workflows/` - Workflow definitions
- `ui/` - Streamlit interface
- `config/` - Configuration files
- `scripts/` - Executable scripts

## Quick Start

### Local Development (CPU)

```bash
# Create CPU virtual environment
python -m venv venv-cpu
source venv-cpu/bin/activate  # On Windows: venv-cpu\Scripts\activate

# Install CPU-only dependencies
pip install -r requirements-cpu.txt

# Run pipeline
python -m agent.main \
  --data_path "your-dataset" \
  --out_dir "./output/run1" \
  --max_iters 3

# Or use UI
streamlit run ui/app.py
```

### Production (GPU)

```bash
# Create GPU virtual environment
python -m venv venv-gpu
source venv-gpu/bin/activate  # On Windows: venv-gpu\Scripts\activate

# Install GPU dependencies
pip install -r requirements.txt

# Run pipeline
python -m agent.main \
  --data_path "your-dataset" \
  --out_dir "./output/run1" \
  --max_iters 3
```

## Supported Training Methods

- SFT (Supervised Fine-Tuning)
- GRPO (Group Relative Policy Optimization)
- DPO (Direct Preference Optimization)
