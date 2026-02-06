"""Pydantic schemas for SFT annotation outputs.

Image annotations use the hand-crafted schema in the prompt (5 VQA pairs).
Text annotations use auto-generated format_instructions from these schemas.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, conlist


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE ANNOTATION (schema matches the image prompt exactly)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageQuality(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW_BLURRY = "Low/Blurry"


class PrimarySubject(str, Enum):
    PERSON_PEOPLE = "Person/People"
    ANIMALS = "Animal(s)"
    VEHICLES = "Vehicle(s)"
    FOOD = "Food"
    OBJECTS = "Object(s)"
    SCENERY_ARCHITECTURE = "Scenery/Architecture"
    OTHER = "Other"


class HumanPresence(str, Enum):
    NONE = "None"
    SINGLE_PERSON = "Single Person"
    SMALL_GROUP = "Small Group (2-5)"
    CROWD = "Crowd (>5)"


class FaceVisibility(str, Enum):
    CLEAR_FACES = "Clear Faces"
    OBSCURED_NO_FACES = "Obscured/No Faces"
    NOT_APPLICABLE = "Not Applicable"


class TextType(str, Enum):
    PRINTED = "Printed"
    HANDWRITTEN = "Handwritten"
    SIGNAGE_LOGO = "Signage/Logo"
    NOT_APPLICABLE = "Not Applicable"


class ContentProperties(BaseModel):
    quality: ImageQuality
    primary_subject: PrimarySubject
    object_count: int = Field(..., ge=0)
    human_presence: HumanPresence
    face_visibility: FaceVisibility


class TextProperties(BaseModel):
    contains_text: bool
    is_suitable_for_ocr: bool
    text_type: TextType


class TaskSuitability(BaseModel):
    is_suitable_for_counting: bool
    is_suitable_for_reasoning: bool
    is_suitable_for_multi_step_instruction: bool


class ImageInfo(BaseModel):
    content_properties: ContentProperties
    text_properties: TextProperties
    task_suitability: TaskSuitability


class CaptionItem(BaseModel):
    text: str = Field(..., min_length=1)


class VQAItem(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class OCRItem(BaseModel):
    instruction: Optional[str] = None
    answer: Optional[str] = None


class ReasonItem(BaseModel):
    instruction: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class InstructFollowItem(BaseModel):
    instruction: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class ImageAnnotation(BaseModel):
    """Full annotation produced by the image prompt (5 VQA pairs)."""
    info: ImageInfo
    caption: CaptionItem
    vqa: conlist(VQAItem, min_length=3, max_length=7)
    ocr: OCRItem
    reason: ReasonItem
    instruct_follow: InstructFollowItem


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT ANNOTATION — knowledge distillation (standalone, no source in training)
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeQAItem(BaseModel):
    """Standalone Q&A — question is natural, answer contains all knowledge."""
    question: str = Field(
        ..., min_length=1,
        description="Natural question a curious person would ask. Must be "
                    "fully understandable without any source text.",
    )
    answer: str = Field(
        ..., min_length=1,
        description="Thorough, expert-level answer (3-8 sentences). Must "
                    "embed all relevant facts. No 'according to the text'.",
    )


class DetailedExplanation(BaseModel):
    """In-depth explanation of a key concept — teaches depth."""
    instruction: str = Field(
        ..., min_length=1,
        description="Natural request like 'Explain X in detail' or "
                    "'Describe how Y works and why it matters'.",
    )
    response: str = Field(
        ..., min_length=1,
        description="Comprehensive, multi-paragraph expert explanation "
                    "(5-10 sentences). Structured like an encyclopedia entry.",
    )


class AnalyticalReasoning(BaseModel):
    """Multi-fact inference — teaches reasoning chains."""
    instruction: str = Field(
        ..., min_length=1,
        description="A 'why' or 'what can we conclude' question requiring "
                    "chaining 2+ facts.",
    )
    response: str = Field(
        ..., min_length=1,
        description="Explicit reasoning chain: 'First [fact A]. Second "
                    "[fact B]. Combining these, [conclusion].' (3-6 sentences).",
    )


class ConversationalExchange(BaseModel):
    """Natural 2-turn dialogue — teaches multi-turn ability."""
    opening_question: str = Field(
        ..., min_length=1,
        description="Natural first question someone would ask.",
    )
    opening_response: str = Field(
        ..., min_length=1,
        description="Thorough, helpful answer (3-5 sentences).",
    )
    follow_up_question: str = Field(
        ..., min_length=1,
        description="Genuine follow-up that digs deeper. Must flow "
                    "naturally from the first answer.",
    )
    follow_up_response: str = Field(
        ..., min_length=1,
        description="Detailed response introducing NEW information not "
                    "in the first answer (3-5 sentences).",
    )


class TextMetadata(BaseModel):
    language: str = Field(
        ..., min_length=1,
        description="ISO 639-1 code of the source text (e.g. 'en', 'kk', 'ru').",
    )
    topics: List[str] = Field(
        ..., min_length=1,
        description="3-6 specific entities, events, or concepts from the source.",
    )
    domain: str = Field(
        ..., min_length=1,
        description="One of: culture, history, science, technology, politics, "
                    "economics, art, religion, geography, cuisine, sports, "
                    "education, health, law, other.",
    )


class TextAnnotation(BaseModel):
    """Full text annotation — standalone knowledge distillation.

    Every field produces training pairs where the source text is NOT included.
    Questions are natural. Answers embed all relevant knowledge.
    """
    knowledge_qa: conlist(KnowledgeQAItem, min_length=3, max_length=7) = Field(
        ...,
        description="5 standalone expert Q&A pairs, each testing a different "
                    "knowledge type: factual, conceptual, comparative, causal, applied.",
    )
    detailed_explanation: DetailedExplanation = Field(
        ...,
        description="In-depth explanation of the richest concept from the source.",
    )
    analytical_reasoning: AnalyticalReasoning = Field(
        ...,
        description="Multi-fact reasoning question with explicit inference chain.",
    )
    conversational_exchange: ConversationalExchange = Field(
        ...,
        description="Natural 2-turn conversation with a deeper follow-up.",
    )
    metadata: TextMetadata
