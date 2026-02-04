"""Cluster failures using TF-IDF + KMeans."""

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

    if not failures:
        preview = {"clusters": []}
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
        cluster = clusters.setdefault(label_int, {"count": 0, "examples": []})
        cluster["count"] = int(cluster["count"]) + 1
        if len(cluster["examples"]) < 3:
            cluster["examples"].append(
                {
                    "input": failure.get("input"),
                    "prediction": failure.get("prediction"),
                    "reasons": failure.get("reasons"),
                }
            )

    preview = {"clusters": [{"cluster": int(k), **v} for k, v in clusters.items()]}
    out_path = Path(out_dir) / "cluster_preview.json"
    out_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    return preview
