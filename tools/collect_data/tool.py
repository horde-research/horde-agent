"""
Data collection tool.

Collects text data for specified language from various sources.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.base_tool import BaseTool

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore[assignment]

class CollectDataTool(BaseTool):
    """
    Collects text data for language model training.
    
    Sources:
    - Local files
    - Hugging Face datasets
    - Web scraping
    - API endpoints
    
    Saves raw data to data/raw/{run_id}/
    """
    
    def execute(self, config):
        """
        Collect data based on configuration.
        
        Args:
            config (dict): Data collection config
                - source: str (file path, dataset name, url)
                - size: int (number of samples)
                - language: str (ISO code)
                
        Returns:
            dict: {
                'data_path': str,
                'num_samples': int,
                'metadata': dict
            }
        """

        if not isinstance(config, dict):
            raise TypeError("CollectDataTool.execute expects config as a dict.")

        provider = (config.get("provider") or config.get("source") or "").strip().lower()
        if provider in {"", "kazparserbot", "kaz_parser_bot", "kaz-parser-bot"}:
            return self._collect_with_kazparserbot(config)

        raise ValueError(f"Unsupported collect_data provider/source: {provider!r}")

    def _collect_with_kazparserbot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if Dataset is None:
            raise RuntimeError(
                "datasets is required to save collected data. "
                "Please ensure 'datasets' is installed (it should be in requirements)."
            )

        cached_raw_path = config.get("raw_result_path") or config.get("cached_raw_path")
        result: Any
        run_dir: Path
        raw_path: Path

        if cached_raw_path:
            raw_path = Path(str(cached_raw_path))
            if not raw_path.exists():
                raise FileNotFoundError(f"raw_result_path not found: {raw_path}")
            run_dir = Path(config.get("run_dir") or config.get("out_dir") or raw_path.parent)
            run_dir.mkdir(parents=True, exist_ok=True)
            result = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            keywords = config.get("keywords") or config.get("seed_keywords") or config.get("queries")
            if not keywords or not isinstance(keywords, (list, tuple)) or not all(isinstance(k, str) for k in keywords):
                raise ValueError("kazparserbot collection requires config['keywords'] as a list[str].")

            run_dir = self._resolve_run_dir(config)
            run_dir.mkdir(parents=True, exist_ok=True)

            # Validate env early to fail fast with a clear error.
            missing_env: List[str] = []
            if not os.getenv("SERPER_API_KEY"):
                missing_env.append("SERPER_API_KEY")
            if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
                missing_env.append("OPENAI_API_KEY")
            if missing_env:
                raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_env)}")

            # Import lazily so the rest of the repo can run without this extra dependency.
            from kazparserbot import KazParserBot  # type: ignore

            options = dict(config.get("options") or {})
            # Support both nested options and flat config keys.
            for k in (
                "google_results_to_get_per_query",
                "top_results_to_get",
                "queries_to_generate_per_keyword",
                "collect_imgs_and_context",
            ):
                if k in config and k not in options:
                    options[k] = config[k]

            bot = KazParserBot.from_env()

            with _pushd(run_dir):
                result = bot.scrape_keywords_sync(list(keywords), **options)

            raw_path = run_dir / "kazparserbot_raw.json"
            _write_json(raw_path, result)

        imgs: List[Any] = []
        if isinstance(result, dict):
            imgs = list(result.get("__imgs__", []) or [])

        max_samples: Optional[int] = config.get("size")
        if max_samples is not None:
            try:
                max_samples = int(max_samples)
            except Exception as e:
                raise ValueError("config['size'] must be an int if provided.") from e

        samples = _extract_text_samples(result)
        if not samples:
            # Fallback: keep at least something for downstream steps.
            samples = [json.dumps(result, ensure_ascii=False)]

        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]

        rows = [{"text": s} for s in samples]
        dataset = Dataset.from_list(rows)

        dataset_dir = run_dir / "dataset"
        dataset.save_to_disk(str(dataset_dir))

        return {
            "data_path": str(dataset_dir),
            "num_samples": len(dataset),
            "metadata": {
                "provider": "kazparserbot",
                "run_dir": str(run_dir),
                "raw_result_path": str(raw_path),
                "imgs_dir": str(run_dir / "imgs"),
                "imgs_count": len(imgs),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _resolve_run_dir(self, config: Dict[str, Any]) -> Path:
        run_dir = config.get("run_dir") or config.get("out_dir")
        if run_dir:
            return Path(run_dir)

        run_id = (config.get("run_id") or "run").strip()
        return Path("data") / "raw" / run_id / "collect_data"


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_text_samples(obj: Any) -> List[str]:
    """
    Best-effort extraction of natural-language text snippets from an arbitrary
    nested structure. This keeps CollectDataTool resilient to upstream schema changes.
    """

    texts: List[str] = []
    seen: set[str] = set()

    def add_text(s: str) -> None:
        s = " ".join((s or "").split())
        if not s:
            return
        if s in seen:
            return
        seen.add(s)
        texts.append(s)

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            add_text(x)
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, list):
            for item in x:
                walk(item)
            return
        if isinstance(x, dict):
            # Avoid pulling image metadata into text samples.
            for k, v in x.items():
                if k == "__imgs__":
                    continue
                if isinstance(v, (dict, list)):
                    walk(v)
                    continue
                if isinstance(v, str):
                    key = k.lower()
                    if "url" in key or "href" in key:
                        continue
                    if (
                        key in {
                            "text",
                            "full_text",
                            "snippet",
                            "google_snippet",
                            "content",
                            "body",
                            "title",
                            "description",
                            "context",
                            "summary",
                        }
                        or key.endswith("text")
                        or "snippet" in key
                        or "content" in key
                    ):
                        add_text(v)
            return

    walk(obj)
    return texts
