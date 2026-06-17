"""Action definitions for the v1 agentic full-pipeline controller."""

from __future__ import annotations

from enum import Enum
from typing import List


class ActionType(str, Enum):
    GENERATE_TAXONOMY = "generate_taxonomy"
    COLLECT_DATA = "collect_data"
    ASSESS_COVERAGE_AND_REFINE_QUERIES = "assess_coverage_and_refine_queries"
    BUILD_SFT_DATASET = "build_sft_dataset"
    BUILD_DATASET = "build_dataset"
    TRAIN_MODEL = "train_model"
    EVALUATE_MODEL = "evaluate_model"
    GENERATE_REPORT = "generate_report"
    STOP_SUCCESS = "stop_success"
    STOP_FAILURE = "stop_failure"


FULL_GRAPH_ACTIONS: List[ActionType] = [
    ActionType.GENERATE_TAXONOMY,
    ActionType.COLLECT_DATA,
    ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES,
    ActionType.BUILD_SFT_DATASET,
    ActionType.BUILD_DATASET,
    ActionType.TRAIN_MODEL,
    ActionType.EVALUATE_MODEL,
    ActionType.GENERATE_REPORT,
]


TERMINAL_ACTIONS = {ActionType.STOP_SUCCESS, ActionType.STOP_FAILURE}


def coerce_action_type(value: ActionType | str) -> ActionType:
    if isinstance(value, ActionType):
        return value
    return ActionType(value)
