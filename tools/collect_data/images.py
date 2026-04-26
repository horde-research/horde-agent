"""
Image extraction + download for the data collection pipeline.

Adapted from horde-research/Kaz-Parser-Bot (collect_imgs.py).
For each scraped page, fetches the raw HTML, extracts <img> tags,
downloads images that meet the minimum dimensions, and returns
records carrying the surrounding text context.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from PIL import Image

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 30.0


async def collect_images(
    links: List[str],
    out_dir: Path,
    *,
    concurrency: int = 50,
    context_size: int = 500,
    min_width: int = 300,
    min_height: int = 300,
) -> List[Dict[str, str]]:
    """Fetch each link's HTML, extract images, download those meeting the size threshold.

    Returns a flat list of dicts keyed by 'url', 'img_url', 'context_text_before',
    'context_text_after', and 'file_path'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _process_link(session, sem, link, out_dir, context_size, min_width, min_height)
            for link in links
        ]
        results: List[Dict[str, str]] = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            if i and i % 10 == 0:
                logger.info("Image collection: %d / %d pages", i, len(tasks))
            results.extend(await coro)
    return results


async def _process_link(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    link: str,
    out_dir: Path,
    context_size: int,
    min_width: int,
    min_height: int,
) -> List[Dict[str, str]]:
    async with sem:
        try:
            async with session.get(link) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()
        except Exception as e:
            logger.warning("HTML fetch failed for '%s': %s", link, e)
            return []

    soup = BeautifulSoup(html, "html.parser")
    imgs = soup.find_all("img")

    markers: Dict[int, str] = {}
    for idx, img in enumerate(imgs):
        marker = f"__IMG_MARKER_{idx}__"
        markers[idx] = marker
        img.insert_before(marker)
    full_text = soup.get_text(" ", strip=True)

    output: List[Dict[str, str]] = []
    for idx, img in enumerate(imgs):
        src = img.get("src")
        if not src:
            continue

        width_attr = img.get("width")
        height_attr = img.get("height")
        if width_attr and height_attr:
            try:
                if int(width_attr) < min_width or int(height_attr) < min_height:
                    continue
            except ValueError:
                pass

        before, after = _extract_context(full_text, markers[idx], context_size)
        file_path = await _download_image(session, sem, src, link, out_dir, min_width, min_height)
        if not file_path:
            continue
        output.append(
            {
                "url": link,
                "img_url": src,
                "context_text_before": before,
                "context_text_after": after,
                "file_path": file_path,
            }
        )
    return output


def _extract_context(full_text: str, marker: str, context_size: int) -> tuple[str, str]:
    idx = full_text.find(marker)
    if idx == -1:
        return "", ""
    before = full_text[max(0, idx - context_size) : idx]
    after = full_text[idx + len(marker) : idx + len(marker) + context_size]
    return before, after


async def _download_image(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    img_url: str,
    base_url: str,
    out_dir: Path,
    min_width: int,
    min_height: int,
) -> str:
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = urljoin(base_url, img_url)
    elif not img_url.startswith(("http://", "https://")):
        img_url = urljoin(base_url, img_url)

    filename = out_dir / f"{hashlib.sha1(img_url.encode('utf-8')).hexdigest()[:16]}.jpg"
    if filename.exists():
        return str(filename)

    async with sem:
        try:
            async with session.get(img_url) as resp:
                if resp.status != 200:
                    return ""
                content = await resp.read()
        except Exception as e:
            logger.debug("Image fetch failed for '%s': %s", img_url, e)
            return ""

    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return ""
        filename.write_bytes(content)
    except Exception as e:
        logger.debug("Image decode failed for '%s': %s", img_url, e)
        return ""
    return str(filename)
