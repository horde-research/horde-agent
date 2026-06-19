import json
from pathlib import Path
from typing import Iterable, List

from .types import ImageItem, TextItem


def load_images(input_dir: str, exts: Iterable[str]) -> List[ImageItem]:
    images: List[ImageItem] = []
    input_path = Path(input_dir)
    ext_set = {ext.lower() for ext in exts}
    for path in input_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            topic_hint = path.parent.name
            images.append(
                ImageItem(
                    item_id=str(path),
                    image_path=str(path),
                    topic_hint=topic_hint,
                    group_key=str(path),
                )
            )
    return images


def load_images_from_manifest(manifest_path: str, exts: Iterable[str] | None = None) -> List[ImageItem]:
    images: List[ImageItem] = []
    ext_set = {ext.lower() for ext in (exts or [".jpg", ".jpeg", ".png", ".webp"])}
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = payload.get("images", []) if isinstance(payload, dict) else payload

    for record in records:
        if not isinstance(record, dict):
            continue
        file_path = str(record.get("file_path") or "").strip()
        if not file_path:
            continue
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() not in ext_set:
            continue
        images.append(
            ImageItem(
                item_id=str(path),
                image_path=str(path),
                topic_hint=_image_topic_hint(record),
                group_key=_image_group_key(record, path),
                source_url=str(record.get("url") or "").strip() or None,
                source_image_url=str(record.get("img_url") or "").strip() or None,
                source_query=str(record.get("query") or "").strip() or None,
                source_excerpt=_image_source_excerpt(record),
            )
        )
    return images


def _image_topic_hint(record: dict) -> str:
    domain = str(record.get("domain_label") or record.get("domain_id") or "").strip()
    subdomain = str(record.get("subdomain_label") or record.get("subdomain_id") or "").strip()
    query = str(record.get("query") or "").strip()
    parts = [part for part in (domain, subdomain, query) if part]
    return " / ".join(parts)


def _image_group_key(record: dict, path: Path) -> str:
    cluster_id = str(record.get("dedup_cluster_id") or "").strip()
    if cluster_id:
        return f"image_cluster:{cluster_id}"
    return str(path)


def _image_source_excerpt(record: dict) -> str | None:
    before = str(record.get("context_text_before") or "").strip()
    after = str(record.get("context_text_after") or "").strip()
    excerpt = " ".join(part for part in (before, after) if part).strip()
    return excerpt or None


def load_texts_from_dir(input_dir: str) -> List[TextItem]:
    items: List[TextItem] = []
    for path in Path(input_dir).rglob("*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            items.append(TextItem(item_id=str(path), text=text))
    return items


def load_texts_from_jsonl(input_jsonl: str, text_field: str) -> List[TextItem]:
    items: List[TextItem] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = str(payload.get(text_field, "")).strip()
            if not text:
                continue
            item_id = payload.get("id") or payload.get("source_id") or f"line_{i}"
            source_id = str(payload.get("source_id") or item_id).strip()
            source_url = str(payload.get("source_url") or payload.get("url") or "").strip()
            source_excerpt = str(payload.get("source_excerpt") or text[:2000]).strip()
            group_key = str(payload.get("group_key") or source_url or source_id or item_id).strip()
            items.append(
                TextItem(
                    item_id=str(item_id),
                    text=text,
                    group_key=group_key or str(item_id),
                    source_url=source_url or None,
                    source_id=source_id or None,
                    source_query=str(payload.get("source_query") or payload.get("query") or "").strip() or None,
                    source_excerpt=source_excerpt or None,
                )
            )
    return items
