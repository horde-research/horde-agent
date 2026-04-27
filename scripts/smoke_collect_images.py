"""Smoke test for tools/collect_data/images.collect_images.

Run from repo root:
    python scripts/smoke_collect_images.py
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from tools.collect_data.images import collect_images

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

URLS = [
    "https://www.gnu.org/home.en.html",
    "https://docs.python.org/3/",
    "https://www.python.org",
]

OUT = Path("/tmp/horde_image_smoke")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = await collect_images(
        URLS,
        OUT,
        concurrency=10,
        context_size=200,
        min_width=100,
        min_height=100,
    )
    print(f"\n=== {len(records)} images saved to {OUT} ===")
    for r in records[:5]:
        print(json.dumps({k: r[k][:80] if isinstance(r[k], str) else r[k] for k in r}, ensure_ascii=False, indent=2))
    index = OUT / "index.json"
    index.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nIndex written to {index}")


if __name__ == "__main__":
    asyncio.run(main())
