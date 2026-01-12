"""Statistics calculations for chat history."""

from collections import defaultdict
from datetime import date

import tiktoken

from .db import Database
from .models import DailyStats


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate
        return len(text) // 4


def calculate_daily_stats(db: Database) -> dict[date, DailyStats]:
    """Calculate statistics aggregated by day."""
    daily_data: dict[date, dict] = defaultdict(
        lambda: {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "user_tokens": 0,
            "assistant_tokens": 0,
            "conversations": set(),
        }
    )

    for msg in db.get_all_messages():
        msg_date = msg["date"]
        if isinstance(msg_date, str):
            msg_date = date.fromisoformat(msg_date)

        data = daily_data[msg_date]
        data["total_messages"] += 1
        data["conversations"].add(msg["conversation_id"])

        tokens = msg["token_count"] or count_tokens(msg["content"])

        if msg["role"] == "user":
            data["user_messages"] += 1
            data["user_tokens"] += tokens
        else:
            data["assistant_messages"] += 1
            data["assistant_tokens"] += tokens

    return {
        d: DailyStats(
            date=d,
            total_messages=data["total_messages"],
            user_messages=data["user_messages"],
            assistant_messages=data["assistant_messages"],
            user_tokens=data["user_tokens"],
            assistant_tokens=data["assistant_tokens"],
            conversation_count=len(data["conversations"]),
        )
        for d, data in daily_data.items()
    }


def compute_and_store_stats(db: Database):
    """Compute all stats and store in database."""
    daily = calculate_daily_stats(db)
    for stats in daily.values():
        db.update_daily_stats(stats)


def get_summary_stats(db: Database) -> dict:
    """Get high-level summary statistics."""
    daily_stats = db.get_daily_stats()

    if not daily_stats:
        return {
            "total_days": 0,
            "total_messages": 0,
            "total_user_tokens": 0,
            "total_assistant_tokens": 0,
            "total_conversations": 0,
        }

    return {
        "total_days": len(daily_stats),
        "total_messages": sum(s.total_messages for s in daily_stats),
        "total_user_tokens": sum(s.user_tokens for s in daily_stats),
        "total_assistant_tokens": sum(s.assistant_tokens for s in daily_stats),
        "total_conversations": sum(s.conversation_count for s in daily_stats),
        "first_date": min(s.date for s in daily_stats),
        "last_date": max(s.date for s in daily_stats),
        "avg_messages_per_day": sum(s.total_messages for s in daily_stats) / len(daily_stats),
        "most_active_day": max(daily_stats, key=lambda s: s.total_messages),
    }


def get_top_days(db: Database, limit: int = 10) -> list[DailyStats]:
    """Get the most active days by message count."""
    daily_stats = db.get_daily_stats()
    return sorted(daily_stats, key=lambda s: s.total_messages, reverse=True)[:limit]


def get_model_distribution(db: Database) -> dict[str, int]:
    """Get distribution of messages by model."""
    if not db.conn:
        raise RuntimeError("Database not connected")

    cursor = db.conn.execute("""
        SELECT c.model, COUNT(m.id) as count
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.model IS NOT NULL
        GROUP BY c.model
        ORDER BY count DESC
    """)
    return {row["model"]: row["count"] for row in cursor}


def get_topic_distribution(db: Database) -> dict[str, int]:
    """Get distribution of conversations by topic."""
    if not db.conn:
        raise RuntimeError("Database not connected")

    cursor = db.conn.execute("""
        SELECT topic, COUNT(*) as count
        FROM conversations
        WHERE topic IS NOT NULL
        GROUP BY topic
        ORDER BY count DESC
    """)
    return {row["topic"]: row["count"] for row in cursor}
