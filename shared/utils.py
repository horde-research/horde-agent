"""
Common utility functions.

Shared across all tools and agent components.
"""

import os
import json
from datetime import datetime

def generate_run_id():
    """Generate unique run identifier."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    """Save dict to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """Load JSON file to dict."""
    with open(path, 'r') as f:
        return json.load(f)
