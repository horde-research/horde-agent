"""Image deduplication for local image manifests.

This mirrors the small SSCD flow used by Hugging Face's image dedup toolkit:
load a TorchScript SSCD model, resize RGB images to 320x320, normalize with
ImageNet stats, compute embeddings, and cluster records whose cosine similarity
is above a configured threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageFile


logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_SSCD_MODEL_PATH = "models/sscd_disc_mixup.torchscript.pt"
DEFAULT_SSCD_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/"
    "sscd_disc_mixup.torchscript.pt"
)

EmbeddingFn = Callable[[list[Path]], list[list[float]]]


def deduplicate_image_records(
    records: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    threshold: float = 0.90,
    model_path: str | Path = DEFAULT_SSCD_MODEL_PATH,
    model_url: str = DEFAULT_SSCD_MODEL_URL,
    batch_size: int = 32,
    max_reported_pairs: int = 100,
    device: str | None = None,
    embedding_fn: EmbeddingFn | None = None,
) -> dict[str, Any]:
    """Return a deduped image manifest and a JSON-serializable report."""
    started_at = time.time()
    rows = [dict(record) for record in records]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_items = _image_items(rows)
    union_find = _UnionFind(len(rows))
    exact = _union_exact_file_duplicates(image_items, union_find, max_reported_pairs=max_reported_pairs)

    embedding = {"pair_count": 0, "pairs": []}
    model_info: dict[str, Any] = {}
    if len(image_items) >= 2:
        paths = [item["path"] for item in image_items]
        if embedding_fn is None:
            embeddings, model_info = _compute_sscd_embeddings(
                paths,
                model_path=model_path,
                model_url=model_url,
                batch_size=batch_size,
                device=device,
            )
        else:
            embeddings = embedding_fn(paths)
            model_info = {
                "model_path": str(model_path),
                "model_url": model_url,
                "device": "test",
                "downloaded": False,
            }
        embedding = _union_embedding_duplicates(
            image_items,
            embeddings,
            union_find,
            threshold=threshold,
            max_reported_pairs=max_reported_pairs,
        )

    clusters = _clusters(rows, image_items, union_find)
    kept_indices, removed_indices, cluster_rows = _select_representatives(rows, image_items, clusters)
    deduped_records = [_annotated_record(rows[idx], cluster_rows.get(idx)) for idx in kept_indices]

    report = {
        "schema_version": "image_dedup.v1",
        "enabled": True,
        "method": "sscd",
        "threshold": float(threshold),
        "model_path": model_info.get("model_path", str(model_path)),
        "model_url": model_info.get("model_url", model_url),
        "device": model_info.get("device"),
        "downloaded_model": model_info.get("downloaded", False),
        "num_input_records": len(rows),
        "num_processable_records": len(image_items),
        "num_missing_or_invalid_paths": len(rows) - len(image_items),
        "num_kept_records": len(deduped_records),
        "num_removed_records": len(removed_indices),
        "num_duplicate_clusters": len([cluster for cluster in clusters if len(cluster) > 1]),
        "exact_duplicate_pair_count": exact["pair_count"],
        "embedding_duplicate_pair_count": embedding["pair_count"],
        "kept_indices": kept_indices,
        "removed_indices": removed_indices,
        "exact_duplicate_pairs": exact["pairs"],
        "embedding_duplicate_pairs": embedding["pairs"],
        "clusters": [
            cluster
            for cluster in cluster_rows.values()
            if cluster and cluster["representative_index"] in kept_indices
        ][:max_reported_pairs],
        "timing": {
            "total_seconds": round(time.time() - started_at, 4),
            **model_info.get("timing", {}),
        },
    }
    return {"records": deduped_records, "report": report}


def write_image_dedup_report(report: dict[str, Any], path: str | Path) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _image_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        file_path = str(record.get("file_path") or "").strip()
        if not file_path:
            continue
        path = Path(file_path)
        if path.exists() and path.is_file():
            items.append({"record_index": idx, "path": path})
    return items


def _union_exact_file_duplicates(
    image_items: list[dict[str, Any]],
    union_find: "_UnionFind",
    *,
    max_reported_pairs: int,
) -> dict[str, Any]:
    seen: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    pair_count = 0
    for item in image_items:
        idx = int(item["record_index"])
        digest = _file_sha256(Path(item["path"]))
        if digest in seen:
            left_idx = seen[digest]
            union_find.union(left_idx, idx)
            pair_count += 1
            if len(pairs) < max_reported_pairs:
                pairs.append(_pair(left_idx, idx, score=1.0, reason="exact_file_hash"))
        else:
            seen[digest] = idx
    return {"pair_count": pair_count, "pairs": pairs}


def _compute_sscd_embeddings(
    paths: list[Path],
    *,
    model_path: str | Path,
    model_url: str,
    batch_size: int,
    device: str | None,
) -> tuple[list[list[float]], dict[str, Any]]:
    import torch

    resolved_model_path, downloaded = ensure_sscd_model(model_path, model_url=model_url)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    load_started = time.time()
    model = torch.jit.load(str(resolved_model_path), map_location=resolved_device)
    model.eval()
    model = model.to(resolved_device)
    load_seconds = time.time() - load_started

    embeddings: list[list[float]] = []
    inference_seconds = 0.0
    actual_batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(paths), actual_batch_size):
            batch_paths = paths[start : start + actual_batch_size]
            batch = torch.stack([_preprocess_image(path) for path in batch_paths]).to(resolved_device)
            infer_started = time.time()
            output = model(batch)
            inference_seconds += time.time() - infer_started
            output = torch.nn.functional.normalize(output.float(), p=2, dim=1)
            embeddings.extend(output.cpu().tolist())

    return embeddings, {
        "model_path": str(resolved_model_path),
        "model_url": model_url,
        "device": resolved_device,
        "downloaded": downloaded,
        "timing": {
            "model_load_seconds": round(load_seconds, 4),
            "model_inference_seconds": round(inference_seconds, 4),
        },
    }


def ensure_sscd_model(
    model_path: str | Path = DEFAULT_SSCD_MODEL_PATH,
    *,
    model_url: str = DEFAULT_SSCD_MODEL_URL,
) -> tuple[Path, bool]:
    """Ensure the TorchScript SSCD model exists locally, downloading if needed."""
    path = Path(model_path).expanduser()
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    logger.info("Downloading SSCD image dedup model from %s to %s", model_url, path)
    try:
        with urllib.request.urlopen(model_url, timeout=120) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path, True


def _preprocess_image(path: Path) -> Any:
    import torch

    with Image.open(path) as image:
        image = image.convert("RGB").resize((320, 320), Image.Resampling.BICUBIC)
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(320, 320, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _union_embedding_duplicates(
    image_items: list[dict[str, Any]],
    embeddings: list[list[float]],
    union_find: "_UnionFind",
    *,
    threshold: float,
    max_reported_pairs: int,
) -> dict[str, Any]:
    if len(embeddings) != len(image_items):
        raise ValueError("Embedding count must match processable image count.")
    if len(embeddings) < 2:
        return {"pair_count": 0, "pairs": []}

    import torch

    vectors = torch.tensor(embeddings, dtype=torch.float32)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    pairs: list[dict[str, Any]] = []
    pair_count = 0
    for i in range(len(image_items)):
        sims = torch.mv(vectors[i + 1 :], vectors[i])
        matches = torch.where(sims >= float(threshold))[0].tolist()
        for offset in matches:
            j = i + 1 + int(offset)
            left_idx = int(image_items[i]["record_index"])
            right_idx = int(image_items[j]["record_index"])
            if union_find.find(left_idx) == union_find.find(right_idx):
                continue
            score = float(sims[int(offset)].item())
            union_find.union(left_idx, right_idx)
            pair_count += 1
            if len(pairs) < max_reported_pairs:
                pairs.append(_pair(left_idx, right_idx, score=score, reason="sscd_embedding"))
    return {"pair_count": pair_count, "pairs": pairs}


def _clusters(
    records: list[dict[str, Any]],
    image_items: list[dict[str, Any]],
    union_find: "_UnionFind",
) -> list[list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    processable_indices = {int(item["record_index"]) for item in image_items}
    for idx in processable_indices:
        grouped[union_find.find(idx)].append(idx)
    for idx in range(len(records)):
        if idx not in processable_indices:
            grouped[idx].append(idx)
    return [sorted(cluster) for cluster in grouped.values()]


def _select_representatives(
    records: list[dict[str, Any]],
    image_items: list[dict[str, Any]],
    clusters: list[list[int]],
) -> tuple[list[int], list[int], dict[int, dict[str, Any]]]:
    path_by_idx = {int(item["record_index"]): Path(item["path"]) for item in image_items}
    kept: list[int] = []
    removed: list[int] = []
    cluster_rows: dict[int, dict[str, Any]] = {}
    cluster_id = 0
    for cluster in clusters:
        representative = max(cluster, key=lambda idx: _record_quality_score(records[idx], path_by_idx.get(idx), idx))
        kept.append(representative)
        duplicate_members = [idx for idx in cluster if idx != representative]
        removed.extend(duplicate_members)
        if duplicate_members:
            row = {
                "cluster_id": cluster_id,
                "representative_index": representative,
                "member_indices": cluster,
                "removed_indices": duplicate_members,
                "cluster_size": len(cluster),
            }
            cluster_rows[representative] = row
            cluster_id += 1
    return sorted(kept), sorted(removed), cluster_rows


def _record_quality_score(record: dict[str, Any], path: Path | None, idx: int) -> tuple[int, int, int]:
    width = _int(record.get("width"), 0)
    height = _int(record.get("height"), 0)
    file_size = path.stat().st_size if path and path.exists() else 0
    return width * height, file_size, -idx


def _annotated_record(record: dict[str, Any], cluster: dict[str, Any] | None) -> dict[str, Any]:
    output = dict(record)
    if cluster:
        output["dedup_cluster_id"] = str(cluster["cluster_id"])
        output["dedup_cluster_size"] = str(cluster["cluster_size"])
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair(left_idx: int, right_idx: int, *, score: float, reason: str) -> dict[str, Any]:
    return {
        "left_index": int(left_idx),
        "right_index": int(right_idx),
        "score": round(float(score), 4) if math.isfinite(float(score)) else 0.0,
        "reason": reason,
    }


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1
