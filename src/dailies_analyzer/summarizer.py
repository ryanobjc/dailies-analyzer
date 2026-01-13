"""Conversation summarization using Claude Opus."""

import json
import os
from datetime import datetime

from anthropic import Anthropic
from rich.console import Console
from rich.progress import Progress

from .db import Database
from .extractor import get_conversation_text, parse_json_response
from .models import ConversationSummary

console = Console()

SUMMARY_PROMPT = """Summarize this conversation between a senior engineer and an AI assistant.

Topic: {topic}
Date: {date}
Message count: {message_count}

Provide a structured summary with:
- summary: 2-4 sentence overview of what was discussed and accomplished
- key_topics: array of 3-6 main topics/technologies discussed
- sentiment: the overall tone (one of: "technical", "exploratory", "debugging", "learning", "planning", "creative", "frustrated", "collaborative")
- outcome: what happened (one of: "resolved", "ongoing", "abandoned", "learning", "decision_made", "idea_generated")

Return ONLY a valid JSON object with these fields, no other text.

<conversation>
{conversation}
</conversation>"""


def summarize_conversation(
    client: Anthropic, conversation_text: str, topic: str | None, date: str | None, message_count: int
) -> dict | None:
    """Summarize a single conversation using Claude Opus."""
    prompt = SUMMARY_PROMPT.format(
        conversation=conversation_text,
        topic=topic or "Unknown",
        date=date or "Unknown",
        message_count=message_count,
    )

    try:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        # Parse JSON - handle both object and array responses
        content = content.strip()

        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

        # Find JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]

        return json.loads(content)

    except json.JSONDecodeError as e:
        console.print(f"[yellow]Failed to parse JSON response: {e}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]API error: {e}[/red]")
        return None


def summarize_conversations(db: Database, conversations: list[dict]):
    """Summarize a list of conversations."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return

    client = Anthropic(api_key=api_key)
    total_summaries = 0

    with Progress() as progress:
        task = progress.add_task("Summarizing conversations...", total=len(conversations))

        for conv in conversations:
            conv_id = conv["id"]
            conv_text = get_conversation_text(db, conv_id)
            topic = conv.get("topic")
            date = conv.get("date")
            message_count = conv.get("message_count", 0)

            # Skip very short conversations
            if len(conv_text) < 200:
                progress.advance(task)
                continue

            # Truncate very long conversations
            if len(conv_text) > 80000:
                conv_text = conv_text[:80000] + "\n[TRUNCATED]"

            summary_data = summarize_conversation(client, conv_text, topic, date, message_count)

            if summary_data:
                try:
                    summary = ConversationSummary(
                        conversation_id=conv_id,
                        summary=summary_data.get("summary", ""),
                        key_topics=summary_data.get("key_topics", []),
                        sentiment=summary_data.get("sentiment", "unknown"),
                        outcome=summary_data.get("outcome", "unknown"),
                        summarized_at=datetime.now(),
                    )
                    db.insert_summary(summary)
                    total_summaries += 1
                except (KeyError, ValueError) as e:
                    console.print(f"[yellow]Skipping malformed summary: {e}[/yellow]")

            progress.advance(task)

    console.print(f"[green]Created {total_summaries} summaries from {len(conversations)} conversations[/green]")
