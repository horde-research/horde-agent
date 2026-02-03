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

```bash
# Install dependencies
pip install -r requirements.txt

# Run workflow
python scripts/run_workflow.py --input "Train SFT model on Kazakh language data"

# Or use UI
streamlit run ui/app.py
```

## Supported Training Methods

- SFT (Supervised Fine-Tuning)
- GRPO (Group Relative Policy Optimization)
- DPO (Direct Preference Optimization)
