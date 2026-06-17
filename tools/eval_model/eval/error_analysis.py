"""Cluster failures using TF-IDF + KMeans.

Copied from `agentic_train_pipeline/eval/error_analysis.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_failures(failures_path: str, out_dir: str) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    with open(failures_path, "r", encoding="utf-8") as handle:
        for line in handle:
            failures.append(json.loads(line))

    out_path = Path(out_dir) / "cluster_preview.json"

    if not failures:
        preview = {"clusters": []}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
        return preview

    if len(failures) < 2:
        failure = failures[0]
        preview = {
            "clusters": [
                {
                    "cluster": 0,
                    "label": failure.get("label") or _label_from_reasons(failure.get("reasons") or []),
                    "count": 1,
                    "examples": [
                        {
                            "input": failure.get("input"),
                            "prediction": failure.get("prediction"),
                            "reasons": failure.get("reasons"),
                        }
                    ],
                }
            ]
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
        return preview

    texts = [f"{row.get('input','')} {row.get('prediction','')}" for row in failures]
    vectorizer = TfidfVectorizer(max_features=2000)
    matrix = vectorizer.fit_transform(texts)

    n_failures = len(failures)
    k = min(8, max(2, n_failures // 10))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(matrix)

    clusters: Dict[int, Dict[str, Any]] = {}
    for label, failure in zip(labels, failures):
        label_int = int(label)
        cluster = clusters.setdefault(label_int, {"count": 0, "examples": [], "labels": {}})
        cluster["count"] = int(cluster["count"]) + 1
        failure_label = failure.get("label") or _label_from_reasons(failure.get("reasons") or [])
        cluster["labels"][failure_label] = int(cluster["labels"].get(failure_label, 0)) + 1
        if len(cluster["examples"]) < 3:
            cluster["examples"].append(
                {
                    "input": failure.get("input"),
                    "prediction": failure.get("prediction"),
                    "reasons": failure.get("reasons"),
                }
            )

    preview_clusters = []
    for key, value in clusters.items():
        labels = value.pop("labels", {})
        top_label = max(labels.items(), key=lambda item: item[1])[0] if labels else "other_failure"
        preview_clusters.append({"cluster": int(key), "label": top_label, **value})
    preview = {"clusters": preview_clusters}
    out_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
    return preview


def _label_from_reasons(reasons: list[str]) -> str:
    joined = " ".join(str(reason) for reason in reasons)
    if "image" in joined:
        return "image_processing_failure"
    if "empty" in joined or "short" in joined:
        return "generation_empty_or_short"
    if "repetition" in joined or "prompt_echo" in joined:
        return "generation_repetition"
    if "similarity" in joined:
        return "semantic_mismatch"
    return "other_failure"
