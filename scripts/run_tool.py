#!/usr/bin/env python3
"""
Run individual tool for testing.

Usage:
    python scripts/run_tool.py --tool collect_data --config config.yaml
"""

import argparse

def main():
    """Execute single tool in isolation."""
    parser = argparse.ArgumentParser(description='Run individual tool')
    parser.add_argument('--tool', type=str, required=True, 
                       choices=['parse_input', 'collect_data', 'eval_data', 
                               'build_dataset', 'train', 'eval_model', 'report'])
    parser.add_argument('--config', type=str, help='Tool configuration')
    
    args = parser.parse_args()
    
    print(f"Running tool: {args.tool}")
    # Tool execution implementation

if __name__ == '__main__':
    main()
