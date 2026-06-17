"""
Serper image-search collector for the data collection pipeline.

This collector searches Google Images through Serper, downloads candidate
images directly, validates image dimensions, and returns an index compatible
with the existing image SFT pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote, urlparse

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)

SERPER_IMAGES_URL = "https://google.serper.dev/images"
HTTP_TIMEOUT_S = 30.0

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


async def collect_images_from_serper(
    queries: List[Union[str, Dict[str, str]]],
    out_dir: Path,
    *,
    serper_key: str,
    results_per_query: int = 10,
    concurrency: int = 50,
    min_width: int = 300,
    min_height: int = 300,
) -> List[Dict[str, str]]:
    """Search Serper Images for each query and download usable images.

    Returned records include the existing keys expected downstream:
    ``url``, ``img_url``, and ``file_path``. Extra metadata is preserved for
    debugging and LangSmith traces: query, title, source, dimensions, thumbnail.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    query_specs = _normalize_query_specs(queries)
    if not query_specs:
        return []

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        search_tasks = [
            _search_images(
                session=session,
                sem=sem,
                query_spec=query_spec,
                serper_key=serper_key,
                results_per_query=max(1, int(results_per_query)),
            )
            for query_spec in query_specs
        ]

        candidates: List[Dict[str, str]] = []
        seen_urls: set[str] = set()
        for coro in asyncio.as_completed(search_tasks):
            query_spec, entries = await coro
            for entry in entries:
                record = _normalize_image_record(query_spec, entry)
                if not record:
                    continue
                img_url = record["img_url"]
                if img_url in seen_urls:
                    continue
                seen_urls.add(img_url)
                candidates.append(record)

        download_tasks = [
            _download_image_record(
                session=session,
                sem=sem,
                record=record,
                out_dir=out_dir,
                min_width=min_width,
                min_height=min_height,
            )
            for record in candidates
        ]

        records: List[Dict[str, str]] = []
        for i, coro in enumerate(asyncio.as_completed(download_tasks)):
            if i and i % 25 == 0:
                logger.info("Serper image downloads: %d / %d candidates", i, len(download_tasks))
            downloaded = await coro
            if downloaded:
                records.append(downloaded)
        return records


async def _search_images(
    *,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    query_spec: Dict[str, str],
    serper_key: str,
    results_per_query: int,
) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    query = query_spec["query"]
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    body = {"q": query, "num": results_per_query}
    async with sem:
        try:
            async with session.post(SERPER_IMAGES_URL, json=body, headers=headers) as resp:
                if resp.status >= 400:
                    logger.warning("Serper image search failed for '%s': HTTP %s", query[:60], resp.status)
                    return query_spec, []
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("Serper image search failed for '%s': %s", query[:60], exc)
            return query_spec, []
    return query_spec, list(data.get("images") or [])


def _normalize_image_record(query: Union[str, Dict[str, str]], entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    query_spec = _normalize_query_spec(query)
    img_url = _select_image_url(entry)
    if not img_url:
        return None
    record = {
        key: str(value)
        for key, value in query_spec.items()
        if value is not None and str(value).strip()
    }
    record.update(
        {
            "query": query_spec["query"],
            "title": str(entry.get("title") or ""),
            "source": str(entry.get("source") or ""),
            "url": str(entry.get("link") or entry.get("sourceUrl") or ""),
            "img_url": img_url,
            "thumbnail_url": str(entry.get("thumbnailUrl") or ""),
        }
    )
    return record


def _normalize_query_specs(queries: List[Union[str, Dict[str, str]]]) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    for query in queries:
        spec = _normalize_query_spec(query)
        if spec["query"]:
            specs.append(spec)
    return specs


def _normalize_query_spec(query: Union[str, Dict[str, str]]) -> Dict[str, str]:
    if isinstance(query, dict):
        normalized = {
            str(key): str(value)
            for key, value in query.items()
            if value is not None
        }
        normalized["query"] = str(normalized.get("query") or "").strip()
        return normalized
    return {
        "query": str(query or "").strip(),
    }


def _select_image_url(entry: Dict[str, Any]) -> str:
    for key in ("imageUrl", "image_url", "thumbnailUrl", "thumbnail_url"):
        value = entry.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return ""


async def _download_image_record(
    *,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    record: Dict[str, str],
    out_dir: Path,
    min_width: int,
    min_height: int,
) -> Optional[Dict[str, str]]:
    img_url = record["img_url"]
    stem = hashlib.sha1(img_url.encode("utf-8")).hexdigest()[:16]
    existing = _find_existing_image(out_dir, stem)
    if existing:
        return {**record, "file_path": str(existing)}

    async with sem:
        try:
            async with session.get(img_url) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                content = await resp.read()
        except Exception as exc:
            logger.debug("Image fetch failed for '%s': %s", img_url, exc)
            return None

    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return None
    except Exception as exc:
        logger.debug("Image decode failed for '%s': %s", img_url, exc)
        return None

    suffix = _image_extension_from_content_type(content_type) or _image_extension_from_url(img_url) or ".jpg"
    file_path = out_dir / f"{stem}{suffix}"
    file_path.write_bytes(content)
    return {
        **record,
        "file_path": str(file_path),
        "width": str(width),
        "height": str(height),
    }


def _find_existing_image(out_dir: Path, stem: str) -> Optional[Path]:
    for suffix in _IMAGE_EXTENSIONS:
        path = out_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _image_extension_from_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return _CONTENT_TYPE_EXTENSIONS.get(media_type, "")


def _image_extension_from_url(url: str) -> str:
    try:
        suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    except Exception:
        return ""
    return suffix if suffix in _IMAGE_EXTENSIONS else ""
