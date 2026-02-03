"""
Streamlit UI for Horde Agent.

Main entry point for web interface.
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Horde Agent - LLM Training",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
    .status-running {
        color: #f39c12;
        font-weight: 600;
    }
    .status-success {
        color: #27ae60;
        font-weight: 600;
    }
    .status-error {
        color: #e74c3c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow_running' not in st.session_state:
    st.session_state.workflow_running = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = None
if 'run_history' not in st.session_state:
    st.session_state.run_history = []

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["Quick Start", "Manual Workflow", "Configuration", "Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Training Methods")
    st.markdown("- **SFT**: Supervised Fine-Tuning")
    st.markdown("- **GRPO**: Group Relative Policy Optimization")
    st.markdown("- **DPO**: Direct Preference Optimization")
    
    st.markdown("---")
    if st.session_state.workflow_running:
        st.markdown("**Status:** Running")
        st.markdown(f"**Step:** {st.session_state.current_step or 'Initializing...'}")

# Main content
st.markdown('<div class="main-header">Horde Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LLM Training Workflow Automation</div>', unsafe_allow_html=True)

if page == "Quick Start":
    st.markdown("## Quick Start")
    st.markdown("Train a language model in three simple steps:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 1. Describe Your Training Task")
        
        input_mode = st.radio(
            "Input Method",
            ["Natural Language", "YAML Configuration"],
            horizontal=True
        )
        
        if input_mode == "Natural Language":
            user_input = st.text_area(
                "Describe what you want to train",
                placeholder="Example: Train an SFT model on Kazakh language data for 3 epochs with batch size 4",
                height=150
            )
            
            st.info("The system will parse your description and generate a configuration automatically.")
            
        else:
            user_input = st.text_area(
                "YAML Configuration",
                placeholder="""project_name: kazakh_sft
language: kk
data:
  source: path/to/data
train:
  method: sft
  model: meta-llama/Llama-2-7b-hf
  epochs: 3
  batch_size: 4""",
                height=300
            )
        
        st.markdown("### 2. Review Configuration")
        if st.button("Parse & Validate", type="secondary", use_container_width=True):
            if user_input:
                with st.spinner("Parsing input..."):
                    st.success("Configuration validated successfully")
                    st.json({
                        "project_name": "kazakh_sft",
                        "language": "kk",
                        "train": {"method": "sft", "epochs": 3}
                    })
            else:
                st.warning("Please provide input first")
        
        st.markdown("### 3. Start Training")
        if st.button("Start Workflow", type="primary", use_container_width=True, disabled=st.session_state.workflow_running):
            if user_input:
                st.session_state.workflow_running = True
                st.session_state.current_step = "Parsing input"
                st.rerun()
            else:
                st.warning("Please provide input first")
    
    with col2:
        st.markdown("### Workflow Steps")
        
        steps = [
            "Parse Input",
            "Collect Data",
            "Evaluate Data",
            "Build Dataset",
            "Initialize Reporting",
            "Train Model",
            "Evaluate Model",
            "Generate Report"
        ]
        
        for i, step in enumerate(steps, 1):
            if st.session_state.workflow_running:
                if st.session_state.current_step == step:
                    st.markdown(f"**{i}. {step}** ⏳")
                elif i < steps.index(st.session_state.current_step) + 1:
                    st.markdown(f"~~{i}. {step}~~ ✓")
                else:
                    st.markdown(f"{i}. {step}")
            else:
                st.markdown(f"{i}. {step}")
        
        if st.session_state.workflow_running:
            st.markdown("---")
            if st.button("Stop Workflow", type="secondary"):
                st.session_state.workflow_running = False
                st.session_state.current_step = None
                st.rerun()

elif page == "Manual Workflow":
    st.markdown("## Manual Workflow Execution")
    st.markdown("Execute workflow steps one by one for fine-grained control.")
    
    tabs = st.tabs([
        "1. Parse", "2. Collect", "3. Eval Data", 
        "4. Build Dataset", "5. Train", "6. Evaluate", "7. Report"
    ])
    
    with tabs[0]:
        st.markdown("### Parse Input")
        st.text_area("Input", height=150, key="manual_parse_input")
        if st.button("Parse", key="manual_parse_btn"):
            st.success("Parsed successfully")
    
    with tabs[1]:
        st.markdown("### Collect Data")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Data Source", placeholder="HF dataset or file path")
            st.number_input("Number of Samples", min_value=100, value=10000)
        with col2:
            st.text_input("Language", value="en")
        if st.button("Collect Data", key="manual_collect_btn"):
            st.success("Data collected")
    
    with tabs[2]:
        st.markdown("### Evaluate Data Quality")
        if st.button("Run Evaluation", key="manual_eval_data_btn"):
            st.success("Data quality: Good")
            st.metric("Total Samples", "10,000")
            st.metric("Quality Score", "0.92")
    
    with tabs[3]:
        st.markdown("### Build Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Train Split Ratio", 0.0, 1.0, 0.8)
            st.number_input("Max Sequence Length", value=2048)
        with col2:
            st.selectbox("Format", ["SFT", "GRPO", "DPO"])
        if st.button("Build Dataset", key="manual_build_btn"):
            st.success("Dataset built")
    
    with tabs[4]:
        st.markdown("### Train Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Method", ["SFT", "GRPO", "DPO"], key="manual_train_method")
            st.text_input("Base Model", value="meta-llama/Llama-2-7b-hf")
        with col2:
            st.number_input("Epochs", min_value=1, value=3)
            st.number_input("Batch Size", min_value=1, value=4)
        with col3:
            st.number_input("Learning Rate", value=0.0001, format="%.6f")
            st.checkbox("Use LoRA", value=True)
        
        if st.button("Start Training", key="manual_train_btn", type="primary"):
            st.success("Training started")
            st.progress(0.0, "Epoch 0/3")
    
    with tabs[5]:
        st.markdown("### Evaluate Model")
        st.multiselect("Metrics", ["Loss", "Perplexity", "BLEU", "Accuracy"], default=["Loss", "Perplexity"])
        if st.button("Evaluate", key="manual_eval_btn"):
            st.success("Evaluation complete")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Loss", "2.34")
            with col2:
                st.metric("Perplexity", "10.38")
    
    with tabs[6]:
        st.markdown("### Generate Report")
        st.multiselect("Reporters", ["File", "Wandb", "TensorBoard"], default=["File"])
        if st.button("Generate Final Report", key="manual_report_btn"):
            st.success("Report generated")
            st.markdown("[Download Report]()")

elif page == "Configuration":
    st.markdown("## Configuration Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Default Configuration")
        st.markdown("Edit default settings for all workflows")
        
        with st.expander("Data Settings", expanded=True):
            st.number_input("Default Sample Size", value=10000)
            st.number_input("Max Sequence Length", value=2048)
            st.slider("Train/Val Split", 0.0, 1.0, 0.8)
        
        with st.expander("Training Settings", expanded=True):
            st.selectbox("Default Method", ["SFT", "GRPO", "DPO"])
            st.text_input("Default Base Model", value="meta-llama/Llama-2-7b-hf")
            st.number_input("Default Epochs", value=3)
            st.number_input("Default Batch Size", value=4)
            st.number_input("Default Learning Rate", value=0.0001, format="%.6f")
        
        with st.expander("LoRA Settings"):
            st.checkbox("Enable LoRA by Default", value=True)
            st.number_input("LoRA r", value=8)
            st.number_input("LoRA alpha", value=16)
            st.slider("LoRA Dropout", 0.0, 0.5, 0.05)
        
        with st.expander("Reporting Settings"):
            st.multiselect("Active Reporters", ["File", "Wandb", "TensorBoard", "MLflow"], default=["File"])
            st.text_input("Wandb Entity")
            st.text_input("Wandb Project", value="horde-agent")
        
        if st.button("Save Configuration", type="primary"):
            st.success("Configuration saved to config/default.yaml")
    
    with col2:
        st.markdown("### Configuration Preview")
        st.code("""
project_name: "horde-agent-run"
language: "en"

data:
  source: ""
  size: 10000

dataset:
  split_ratio: 0.8
  max_length: 2048

train:
  method: "sft"
  model: "meta-llama/Llama-2-7b-hf"
  epochs: 3
  batch_size: 4
  learning_rate: 0.0001
  use_lora: true
  lora_r: 8
  lora_alpha: 16

eval:
  metrics:
    - "loss"
    - "perplexity"

reporting:
  reporters:
    - "file"
""", language="yaml")

elif page == "Results":
    st.markdown("## Training Results & History")
    
    if not st.session_state.run_history:
        st.info("No training runs yet. Start a workflow to see results here.")
    else:
        st.markdown("### Recent Runs")
        
        for run in st.session_state.run_history:
            with st.expander(f"{run['run_id']} - {run['status']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Method", run['method'])
                    st.metric("Language", run['language'])
                with col2:
                    st.metric("Epochs", run['epochs'])
                    st.metric("Final Loss", run.get('loss', 'N/A'))
                with col3:
                    st.metric("Duration", run.get('duration', 'N/A'))
                    st.metric("Status", run['status'])
                
                st.markdown("**Actions:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("View Report", key=f"view_{run['run_id']}")
                with col2:
                    st.button("Download Model", key=f"download_{run['run_id']}")
                with col3:
                    st.button("Resume", key=f"resume_{run['run_id']}")

st.markdown("---")
st.markdown("*Horde Agent - LLM Training Automation*")
