"""LLM-powered insight extraction using Claude API."""

import json
import os
from datetime import datetime

from anthropic import Anthropic
from rich.console import Console
from rich.progress import Progress

from .db import Database
from .models import Insight

console = Console()

EXTRACTION_PROMPT = """Analyze this conversation between a senior engineer (25+ years experience) and an AI assistant.

Topic: {topic}
Date: {date}

Extract valuable insights in these categories:
1. **Wisdom Nuggets**: Hard-won insights, mental models, counterintuitive learnings, architectural principles
2. **Product Ideas**: Business opportunities, feature ideas, market gaps, tool ideas
3. **Programming Tips**: Code patterns, debugging techniques, tool recommendations, best practices
4. **Questions Worth Revisiting**: Deep questions that deserve more exploration, unresolved problems

For each insight found, provide:
- category: one of ["wisdom", "product_idea", "programming_tip", "question"]
- title: concise title (5-10 words)
- summary: 2-3 sentence explanation of the insight
- tags: relevant keywords (3-5 tags)
- confidence: 0.0-1.0 how clearly this is a valuable, actionable insight

Focus on:
- Insights that show experienced engineering judgment
- Non-obvious solutions or approaches
- Reusable patterns or principles
- Ideas with real-world applicability

Skip generic advice or basic information. Return ONLY a valid JSON array, no other text. If no valuable insights, return [].

<conversation>
{conversation}
</conversation>"""


def get_conversation_text(db: Database, conversation_id: int) -> str:
    """Get formatted conversation text for a conversation ID."""
    cursor = db.conn.execute(
        """
        SELECT role, content FROM messages
        WHERE conversation_id = ?
        ORDER BY id
        """,
        (conversation_id,),
    )

    lines = []
    for row in cursor:
        role = "USER" if row["role"] == "user" else "ASSISTANT"
        lines.append(f"[{role}]\n{row['content']}\n")

    return "\n".join(lines)


def parse_json_response(content: str) -> list[dict]:
    """Parse JSON from Claude's response, handling various formats."""
    content = content.strip()

    # Handle markdown code blocks
    if "```" in content:
        # Extract content between first pair of ```
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            # Remove language identifier if present
            if content.startswith("json"):
                content = content[4:]
            elif content.startswith("\n"):
                content = content[1:]
            content = content.strip()

    # Try to find JSON array in the content
    # Look for the first [ and last ]
    start = content.find("[")
    end = content.rfind("]")

    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]

    return json.loads(content)


def extract_from_conversation(
    client: Anthropic, conversation_text: str, topic: str | None, date: str | None
) -> list[dict]:
    """Extract insights from a single conversation using Claude."""
    prompt = EXTRACTION_PROMPT.format(
        conversation=conversation_text,
        topic=topic or "Unknown",
        date=date or "Unknown",
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        insights = parse_json_response(content)
        return insights if isinstance(insights, list) else []

    except json.JSONDecodeError as e:
        console.print(f"[yellow]Failed to parse JSON response: {e}[/yellow]")
        return []
    except Exception as e:
        console.print(f"[red]API error: {e}[/red]")
        return []


def extract_insights(db: Database, conversations: list[dict]):
    """Extract insights from a list of conversations."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return

    client = Anthropic(api_key=api_key)
    total_insights = 0

    with Progress() as progress:
        task = progress.add_task("Extracting insights...", total=len(conversations))

        for conv in conversations:
            conv_id = conv["id"]
            conv_text = get_conversation_text(db, conv_id)
            topic = conv.get("topic")
            date = conv.get("date")

            # Skip very short conversations
            if len(conv_text) < 200:
                progress.advance(task)
                continue

            # Truncate very long conversations
            if len(conv_text) > 50000:
                conv_text = conv_text[:50000] + "\n[TRUNCATED]"

            insights_data = extract_from_conversation(client, conv_text, topic, date)

            # Get first message ID for this conversation to link insights
            cursor = db.conn.execute(
                "SELECT id FROM messages WHERE conversation_id = ? LIMIT 1",
                (conv_id,),
            )
            row = cursor.fetchone()
            message_id = row["id"] if row else None

            for data in insights_data:
                try:
                    insight = Insight(
                        message_id=message_id,
                        category=data["category"],
                        title=data["title"],
                        summary=data["summary"],
                        tags=data.get("tags", []),
                        confidence=float(data.get("confidence", 0.5)),
                        extracted_at=datetime.now(),
                    )
                    db.insert_insight(insight)
                    total_insights += 1
                except (KeyError, ValueError) as e:
                    console.print(f"[yellow]Skipping malformed insight: {e}[/yellow]")

            progress.advance(task)

    console.print(f"[green]Extracted {total_insights} insights from {len(conversations)} conversations[/green]")
