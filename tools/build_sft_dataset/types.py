from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageItem:
    item_id: str
    image_path: str
    topic_hint: Optional[str] = None


@dataclass
class TextItem:
    item_id: str
    text: str
