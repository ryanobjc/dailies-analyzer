"""Data models for dailies analyzer."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal


@dataclass
class Message:
    """A single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str
    char_start: int
    char_end: int
    token_count: int = 0


@dataclass
class Conversation:
    """A conversation extracted from an org file."""

    file_path: str
    date: date
    topic: str | None
    model: str | None
    system_prompt: str | None
    messages: list[Message] = field(default_factory=list)


@dataclass
class DailyStats:
    """Aggregated statistics for a single day."""

    date: date
    total_messages: int
    user_messages: int
    assistant_messages: int
    user_tokens: int
    assistant_tokens: int
    conversation_count: int


@dataclass
class Insight:
    """An extracted insight from a conversation."""

    message_id: int
    category: Literal["wisdom", "product_idea", "programming_tip", "question"]
    title: str
    summary: str
    tags: list[str]
    confidence: float
    extracted_at: datetime = field(default_factory=datetime.now)
