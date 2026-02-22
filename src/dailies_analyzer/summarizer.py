"""Conversation summarization using Claude API."""

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
