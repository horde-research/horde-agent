from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageItem:
    item_id: str
    image_path: str
    topic_hint: Optional[str] = None
    group_key: Optional[str] = None
    source_url: Optional[str] = None
    source_image_url: Optional[str] = None
    source_query: Optional[str] = None
    source_excerpt: Optional[str] = None


@dataclass
class TextItem:
    item_id: str
    text: str
    group_key: Optional[str] = None
    source_url: Optional[str] = None
    source_id: Optional[str] = None
    source_query: Optional[str] = None
    source_excerpt: Optional[str] = None
    collection_iteration: Optional[str] = None
