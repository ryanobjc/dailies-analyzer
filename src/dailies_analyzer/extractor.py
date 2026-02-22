"""LLM-powered insight extraction using Claude API."""

import json

from .db import Database

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


