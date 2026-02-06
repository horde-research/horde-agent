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

    if not args.input and not args.config:
        raise SystemExit("Provide either --input or --config")

    # For now, support a minimal config-first entrypoint for the migrated pipeline.
    # Natural language parsing remains a separate concern.
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    else:
        # Accept simple natural language as a stub: treat as dataset id/path.
        cfg = {"data_path": args.input}

    # Defaults
    cfg.setdefault("mode", "workflow")
    cfg.setdefault("run_dir", cfg.get("out_dir") or "./output/run1")
    cfg.setdefault("max_iters", 1)

    from agent.orchestrator import Orchestrator

    result = Orchestrator(cfg).run()
    print(result)

if __name__ == '__main__':
    main()
