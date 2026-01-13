"""SQLite database operations."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from .models import Conversation, ConversationSummary, DailyStats, Insight, Message

EXPORT_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    file_path TEXT,
    date DATE,
    topic TEXT,
    model TEXT,
    message_count INTEGER,
    user_messages INTEGER,
    assistant_messages INTEGER,
    total_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS daily_stats (
    date DATE PRIMARY KEY,
    total_messages INTEGER,
    user_messages INTEGER,
    assistant_messages INTEGER,
    user_tokens INTEGER,
    assistant_tokens INTEGER,
    conversation_count INTEGER
);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    tags TEXT,
    confidence REAL,
    extracted_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tags (
    tag TEXT PRIMARY KEY,
    count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    date DATE,
    topic TEXT,
    model TEXT,
    system_prompt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    token_count INTEGER DEFAULT 0,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS daily_stats (
    date DATE PRIMARY KEY,
    total_messages INTEGER,
    user_messages INTEGER,
    assistant_messages INTEGER,
    user_tokens INTEGER,
    assistant_tokens INTEGER,
    conversation_count INTEGER
);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER,
    category TEXT NOT NULL CHECK (category IN ('wisdom', 'product_idea', 'programming_tip', 'question')),
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    tags TEXT,  -- JSON array
    confidence REAL,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER UNIQUE NOT NULL,
    summary TEXT NOT NULL,
    key_topics TEXT,  -- JSON array
    sentiment TEXT,   -- overall tone: technical, exploratory, frustrated, etc.
    outcome TEXT,     -- resolved, ongoing, abandoned, learning
    summarized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_conversations_date ON conversations(date);
CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
CREATE INDEX IF NOT EXISTS idx_summaries_conversation ON conversation_summaries(conversation_id);
"""


class Database:
    """SQLite database wrapper."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None
        self._has_messages_table: bool | None = None

    def has_messages_table(self) -> bool:
        """Check if this database has a messages table (full DB vs export)."""
        if self._has_messages_table is None:
            if not self.conn:
                raise RuntimeError("Database not connected")
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            )
            self._has_messages_table = cursor.fetchone() is not None
        return self._has_messages_table

    def connect(self) -> sqlite3.Connection:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_schema(self):
        """Initialize the database schema."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def clear_all(self):
        """Clear all data from the database."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        self.conn.executescript("""
            DELETE FROM insights;
            DELETE FROM daily_stats;
            DELETE FROM messages;
            DELETE FROM conversations;
        """)
        self.conn.commit()

    def insert_conversation(self, conv: Conversation) -> int:
        """Insert a conversation and its messages. Returns conversation ID."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            """
            INSERT INTO conversations (file_path, date, topic, model, system_prompt)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conv.file_path, conv.date, conv.topic, conv.model, conv.system_prompt),
        )
        conv_id = cursor.lastrowid

        for msg in conv.messages:
            self.conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, char_start, char_end, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conv_id, msg.role, msg.content, msg.char_start, msg.char_end, msg.token_count),
            )

        return conv_id

    def insert_conversations(self, conversations: list[Conversation]):
        """Batch insert conversations."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        for conv in conversations:
            self.insert_conversation(conv)

        self.conn.commit()

    def get_all_messages(self) -> Iterator[dict]:
        """Get all messages."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute("""
            SELECT m.*, c.date, c.topic, c.model
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            ORDER BY c.date, m.id
        """)
        for row in cursor:
            yield dict(row)

    def get_messages_by_date(self, target_date: date) -> list[dict]:
        """Get all messages for a specific date."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            """
            SELECT m.*, c.date, c.topic, c.model
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.date = ?
            ORDER BY m.id
            """,
            (target_date,),
        )
        return [dict(row) for row in cursor]

    def get_conversation_count(self) -> int:
        """Get total number of conversations."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        cursor = self.conn.execute("SELECT COUNT(*) FROM conversations")
        return cursor.fetchone()[0]

    def get_message_count(self) -> int:
        """Get total number of messages."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        cursor = self.conn.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]

    def update_daily_stats(self, stats: DailyStats):
        """Insert or update daily stats."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        self.conn.execute(
            """
            INSERT OR REPLACE INTO daily_stats
            (date, total_messages, user_messages, assistant_messages,
             user_tokens, assistant_tokens, conversation_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats.date,
                stats.total_messages,
                stats.user_messages,
                stats.assistant_messages,
                stats.user_tokens,
                stats.assistant_tokens,
                stats.conversation_count,
            ),
        )
        self.conn.commit()

    def get_daily_stats(self) -> list[DailyStats]:
        """Get all daily stats."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            "SELECT * FROM daily_stats WHERE date IS NOT NULL ORDER BY date"
        )
        return [
            DailyStats(
                date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                total_messages=row["total_messages"],
                user_messages=row["user_messages"],
                assistant_messages=row["assistant_messages"],
                user_tokens=row["user_tokens"],
                assistant_tokens=row["assistant_tokens"],
                conversation_count=row["conversation_count"],
            )
            for row in cursor
        ]

    def insert_insight(self, insight: Insight) -> int:
        """Insert an insight. Returns insight ID."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            """
            INSERT INTO insights (message_id, category, title, summary, tags, confidence, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                insight.message_id,
                insight.category,
                insight.title,
                insight.summary,
                json.dumps(insight.tags),
                insight.confidence,
                insight.extracted_at,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_insights(
        self, category: str | None = None, ascending: bool = False
    ) -> list[dict]:
        """Get insights, optionally filtered by category."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        order = "ASC" if ascending else "DESC"

        if category:
            cursor = self.conn.execute(
                f"SELECT * FROM insights WHERE category = ? ORDER BY confidence {order}",
                (category,),
            )
        else:
            cursor = self.conn.execute(
                f"SELECT * FROM insights ORDER BY confidence {order}"
            )

        return [dict(row) for row in cursor]

    def get_insight_by_id(self, insight_id: int) -> dict | None:
        """Get a single insight by ID."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            "SELECT * FROM insights WHERE id = ?", (insight_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_insight_with_context(self, insight_id: int) -> dict | None:
        """Get an insight with its linked message and conversation context."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if self.has_messages_table():
            cursor = self.conn.execute(
                """
                SELECT
                    i.*,
                    m.role as message_role,
                    m.content as message_content,
                    c.id as conversation_id,
                    c.topic as conversation_topic,
                    c.date as conversation_date,
                    c.model as conversation_model,
                    c.file_path as conversation_file
                FROM insights i
                LEFT JOIN messages m ON i.message_id = m.id
                LEFT JOIN conversations c ON m.conversation_id = c.id
                WHERE i.id = ?
                """,
                (insight_id,),
            )
        else:
            # Export DB: insights linked directly to conversation_id
            cursor = self.conn.execute(
                """
                SELECT
                    i.*,
                    NULL as message_role,
                    NULL as message_content,
                    i.conversation_id as conversation_id,
                    c.topic as conversation_topic,
                    c.date as conversation_date,
                    c.model as conversation_model,
                    c.file_path as conversation_file
                FROM insights i
                LEFT JOIN conversations c ON i.conversation_id = c.id
                WHERE i.id = ?
                """,
                (insight_id,),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_random_insight(self, category: str | None = None) -> dict | None:
        """Get a random insight, optionally filtered by category."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if category:
            cursor = self.conn.execute(
                "SELECT * FROM insights WHERE category = ? ORDER BY RANDOM() LIMIT 1",
                (category,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM insights ORDER BY RANDOM() LIMIT 1"
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_deep_conversations(self, limit: int = 20) -> list[dict]:
        """Get conversations ranked by depth (message count)."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if self.has_messages_table():
            cursor = self.conn.execute(
                """
                SELECT
                    c.id,
                    c.topic,
                    c.date,
                    c.model,
                    c.file_path,
                    COUNT(m.id) as message_count,
                    SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) as user_messages,
                    SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) as assistant_messages,
                    SUM(CASE WHEN m.role='user' THEN m.token_count ELSE 0 END) as user_tokens,
                    SUM(CASE WHEN m.role='assistant' THEN m.token_count ELSE 0 END) as assistant_tokens,
                    SUM(m.token_count) as total_tokens
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY message_count DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            # Export database has stats pre-computed
            cursor = self.conn.execute(
                """
                SELECT
                    id, topic, date, model, file_path,
                    message_count, user_messages, assistant_messages,
                    total_tokens
                FROM conversations
                ORDER BY message_count DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(row) for row in cursor]

    def get_conversation_messages(self, conversation_id: int) -> list[dict]:
        """Get all messages for a conversation. Returns empty list for export DB."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if not self.has_messages_table():
            return []

        cursor = self.conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        )
        return [dict(row) for row in cursor]

    def get_conversation_by_id(self, conversation_id: int) -> dict | None:
        """Get a conversation by ID with stats."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if self.has_messages_table():
            cursor = self.conn.execute(
                """
                SELECT
                    c.*,
                    COUNT(m.id) as message_count,
                    SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) as user_messages,
                    SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) as assistant_messages,
                    SUM(m.token_count) as total_tokens
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE c.id = ?
                GROUP BY c.id
                """,
                (conversation_id,),
            )
        else:
            # Export database has stats pre-computed in conversations table
            cursor = self.conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_tag_counts(self, limit: int | None = None) -> list[tuple[str, int]]:
        """Get tags sorted by usage count."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        # SQLite JSON extraction - get all tags and count them
        cursor = self.conn.execute("""
            WITH tag_entries AS (
                SELECT json_each.value as tag
                FROM insights, json_each(insights.tags)
                WHERE insights.tags IS NOT NULL AND insights.tags != '[]'
            )
            SELECT tag, COUNT(*) as count
            FROM tag_entries
            GROUP BY tag
            ORDER BY count DESC
        """)
        results = [(row[0], row[1]) for row in cursor]
        return results[:limit] if limit else results

    def get_insights_filtered(
        self,
        tag: str | None = None,
        category: str | None = None,
        ascending: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        """Get insights filtered by tag and/or category."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        order = "ASC" if ascending else "DESC"
        conditions = []
        params = []

        if tag:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.conn.execute(
            f"""
            SELECT * FROM insights
            WHERE {where_clause}
            ORDER BY confidence {order}
            """,
            params,
        )
        results = [dict(row) for row in cursor]
        return results[:limit] if limit else results

    def export_to_file(self, output_path: Path) -> dict:
        """Export database without message content for sharing.

        Returns stats about what was exported.
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        import sqlite3

        # Create new database
        if output_path.exists():
            output_path.unlink()

        export_conn = sqlite3.connect(output_path)
        export_conn.executescript(EXPORT_SCHEMA)

        # Export conversations with stats (no messages)
        cursor = self.conn.execute("""
            SELECT
                c.id,
                c.file_path,
                c.date,
                c.topic,
                c.model,
                COUNT(m.id) as message_count,
                SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) as user_messages,
                SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) as assistant_messages,
                SUM(m.token_count) as total_tokens
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
        """)
        conversations = cursor.fetchall()
        export_conn.executemany(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            conversations
        )

        # Export daily_stats
        cursor = self.conn.execute("SELECT * FROM daily_stats")
        daily_stats = cursor.fetchall()
        export_conn.executemany(
            "INSERT INTO daily_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
            daily_stats
        )

        # Export insights (linked to conversation_id instead of message_id)
        cursor = self.conn.execute("""
            SELECT
                i.id,
                m.conversation_id,
                i.category,
                i.title,
                i.summary,
                i.tags,
                i.confidence,
                i.extracted_at
            FROM insights i
            LEFT JOIN messages m ON i.message_id = m.id
        """)
        insights = cursor.fetchall()
        export_conn.executemany(
            "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            insights
        )

        # Export tag counts
        tag_counts = self.get_tag_counts()
        export_conn.executemany(
            "INSERT INTO tags VALUES (?, ?)",
            tag_counts
        )

        export_conn.commit()
        export_conn.close()

        return {
            "conversations": len(conversations),
            "daily_stats": len(daily_stats),
            "insights": len(insights),
            "tags": len(tag_counts),
        }

    def get_unextracted_conversations(self) -> list[dict]:
        """Get conversations that haven't had insights extracted yet."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute("""
            SELECT c.* FROM conversations c
            WHERE NOT EXISTS (
                SELECT 1 FROM insights i
                JOIN messages m ON i.message_id = m.id
                WHERE m.conversation_id = c.id
            )
            ORDER BY c.date
        """)
        return [dict(row) for row in cursor]

    def insert_summary(self, summary: ConversationSummary) -> int:
        """Insert or update a conversation summary. Returns summary ID."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            """
            INSERT OR REPLACE INTO conversation_summaries
            (conversation_id, summary, key_topics, sentiment, outcome, summarized_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                summary.conversation_id,
                summary.summary,
                json.dumps(summary.key_topics),
                summary.sentiment,
                summary.outcome,
                summary.summarized_at,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def has_summaries_table(self) -> bool:
        """Check if this database has a conversation_summaries table."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_summaries'"
        )
        return cursor.fetchone() is not None

    def get_summary(self, conversation_id: int) -> dict | None:
        """Get summary for a conversation."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if not self.has_summaries_table():
            return None

        cursor = self.conn.execute(
            "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_unsummarized_conversations(self, min_messages: int = 4) -> list[dict]:
        """Get conversations that haven't been summarized yet."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute(
            """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE NOT EXISTS (
                SELECT 1 FROM conversation_summaries s
                WHERE s.conversation_id = c.id
            )
            GROUP BY c.id
            HAVING message_count >= ?
            ORDER BY c.date
            """,
            (min_messages,),
        )
        return [dict(row) for row in cursor]

    def get_summaries_filtered(
        self,
        sentiment: str | None = None,
        outcome: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Get summaries with conversation info, optionally filtered."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if not self.has_summaries_table():
            return []

        conditions = []
        params = []

        if sentiment:
            conditions.append("s.sentiment = ?")
            params.append(sentiment)

        if outcome:
            conditions.append("s.outcome = ?")
            params.append(outcome)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.conn.execute(
            f"""
            SELECT
                s.*,
                c.topic,
                c.date,
                c.model,
                COUNT(m.id) as message_count
            FROM conversation_summaries s
            JOIN conversations c ON s.conversation_id = c.id
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE {where_clause}
            GROUP BY s.id
            ORDER BY c.date DESC
            """,
            params,
        )
        results = [dict(row) for row in cursor]
        return results[:limit] if limit else results

    def get_sentiment_counts(self) -> list[tuple[str, int]]:
        """Get count of summaries by sentiment."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if not self.has_summaries_table():
            return []

        cursor = self.conn.execute("""
            SELECT sentiment, COUNT(*) as count
            FROM conversation_summaries
            GROUP BY sentiment
            ORDER BY count DESC
        """)
        return [(row[0], row[1]) for row in cursor]

    def get_outcome_counts(self) -> list[tuple[str, int]]:
        """Get count of summaries by outcome."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if not self.has_summaries_table():
            return []

        cursor = self.conn.execute("""
            SELECT outcome, COUNT(*) as count
            FROM conversation_summaries
            GROUP BY outcome
            ORDER BY count DESC
        """)
        return [(row[0], row[1]) for row in cursor]

    def get_summary_stats(self) -> dict:
        """Get overall summary statistics."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        if not self.has_summaries_table():
            return {"total": 0, "sentiments": [], "outcomes": []}

        cursor = self.conn.execute("SELECT COUNT(*) FROM conversation_summaries")
        total = cursor.fetchone()[0]

        return {
            "total": total,
            "sentiments": self.get_sentiment_counts(),
            "outcomes": self.get_outcome_counts(),
        }
