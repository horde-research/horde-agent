#!/usr/bin/env python3
"""
Run workflow from command line.

Usage:
    python scripts/run_workflow.py --input "Train SFT model on Kazakh data"
    python scripts/run_workflow.py --config my_config.yaml
"""

import argparse

def main():
    """
    Main entry point for workflow execution.
    
    Accepts either:
    - Natural language input
    - YAML config file
    """
    parser = argparse.ArgumentParser(description='Run LLM training workflow')
    parser.add_argument('--input', type=str, help='Natural language input')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, help='Resume from run_id')
    
    args = parser.parse_args()
    
    # Implementation goes here
    # from workflows.manual_pipeline import run_manual_pipeline
    # result = run_manual_pipeline(user_input)
    
    print("Workflow execution not yet implemented")

if __name__ == '__main__':
    main()
