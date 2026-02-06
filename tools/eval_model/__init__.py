"""Evaluation package.

Avoid importing tool classes at package import time to prevent circular imports.
Import `EvalModelTool` from `tools.eval_model.tool` when needed.
"""

__all__ = []
