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
            )
        )
    return images


def _image_topic_hint(record: dict) -> str:
    domain = str(record.get("domain_label") or record.get("domain_id") or "").strip()
    subdomain = str(record.get("subdomain_label") or record.get("subdomain_id") or "").strip()
    query = str(record.get("query") or "").strip()
    parts = [part for part in (domain, subdomain, query) if part]
    return " / ".join(parts)


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
            items.append(TextItem(item_id=str(item_id), text=text))
    return items
