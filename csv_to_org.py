#!/usr/bin/env python3
"""Convert HistoryExport.csv from Claude iOS app to org-mode files for dailies-analyzer.

Usage:
    python csv_to_org.py HistoryExport.csv [output_directory]

The CSV should have columns: Date, Conversation
Conversations use "Question:" and "AI Response:" as line-level markers.
Output files are named YYYY-MM-DD.org with proper GPTEL_BOUNDS for the parser.
"""

import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_csv_date(date_str: str) -> datetime:
    """Parse date like '1/8/25, 10:16 PM'."""
    return datetime.strptime(date_str.strip(), "%m/%d/%y, %I:%M %p")


def parse_conversation(text: str) -> list[dict]:
    """Parse conversation text with Question:/AI Response: markers into messages."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r"(?m)^(?:Question|AI Response):[ \t]*\n", text)
    markers = re.findall(r"(?m)^(Question|AI Response):[ \t]*$", text)

    messages = []
    for i, marker_type in enumerate(markers):
        role = "user" if marker_type == "Question" else "assistant"
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            messages.append({"role": role, "content": escape_org_headlines(content)})

    return messages


def escape_org_headlines(text: str) -> str:
    """Replace '* ' at line starts with '- ' to avoid org headline conflicts."""
    return re.sub(r"(?m)^\* ", "- ", text)


def markdown_to_org(text: str) -> str:
    """Convert markdown formatting to org-mode."""
    lines = text.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                result.append("#+end_src")
                in_code_block = False
            else:
                lang = stripped[3:].strip()
                result.append(f"#+begin_src {lang}" if lang else "#+begin_src")
                in_code_block = True
            continue

        if in_code_block:
            result.append(line)
            continue

        # Headings: # → ****, ## → *****, etc. (+3 to nest under ***)
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            result.append(f'{"*" * (level + 3)} {heading_text}')
            continue

        # Inline code: `text` → ~text~
        line = re.sub(r"`([^`]+)`", r"~\1~", line)
        # Bold: **text** → *text*
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)
        # Images: ![alt](url) → [[url]]
        line = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[[\2]]", line)
        # Links: [text](url) → [[url][text]]
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"[[\2][\1]]", line)

        result.append(line)

    return "\n".join(result)


def make_topic(messages: list[dict]) -> str:
    """Generate a topic from the first user question."""
    for msg in messages:
        if msg["role"] == "user":
            first_line = msg["content"].split("\n")[0].strip()
            if len(first_line) > 60:
                return first_line[:57] + "..."
            return first_line
    return "Conversation"


def build_org_file(conversations: list[dict]) -> str:
    """Build org-mode file content with correct GPTEL_BOUNDS.

    Each conversation becomes a top-level headline. AI response positions
    are tracked and encoded as Emacs 1-indexed character positions in the
    file-level GPTEL_BOUNDS property.
    """
    # Phase 1: Build body content, tracking AI response positions
    body = ""
    ai_positions = []

    for conv in conversations:
        topic = conv["topic"]

        body += f"* {topic}\n"
        body += ":PROPERTIES:\n"
        body += f":GPTEL_TOPIC: {topic}\n"
        body += ":END:\n\n"

        for msg in conv["messages"]:
            if msg["role"] == "user":
                first_line = msg["content"].split("\n")[0].strip()
                if len(first_line) > 60:
                    first_line = first_line[:57] + "..."
                body += f"** {first_line}\n"
                body += msg["content"] + "\n\n"
            else:
                body += "*** Response\n"
                content = markdown_to_org(msg["content"])
                ai_start = len(body)
                body += content
                ai_end = len(body)
                body += "\n\n"
                ai_positions.append((ai_start, ai_end))

    if not ai_positions:
        return body

    # Phase 2: Iteratively compute GPTEL_BOUNDS.
    # The bounds live in the file-level properties block which precedes the body.
    # Since the bounds string length affects the offset, iterate until stable.
    bounds_str = "((response))"
    for _ in range(10):
        props = f":PROPERTIES:\n:GPTEL_BOUNDS: {bounds_str}\n:END:\n\n"
        offset = len(props)
        # Convert body-relative Python 0-indexed → absolute Emacs 1-indexed
        emacs_pos = [(s + offset + 1, e + offset + 1) for s, e in ai_positions]
        pairs = " ".join(f"({s} {e})" for s, e in emacs_pos)
        new_bounds = f"((response {pairs}))"
        if new_bounds == bounds_str:
            break
        bounds_str = new_bounds

    props = f":PROPERTIES:\n:GPTEL_BOUNDS: {bounds_str}\n:END:\n\n"
    return props + body


def main():
    if len(sys.argv) < 2:
        print("Usage: python csv_to_org.py <HistoryExport.csv> [output_dir]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group conversations by date
    by_date = defaultdict(list)
    for row in rows:
        try:
            dt = parse_csv_date(row["Date"])
        except (ValueError, KeyError) as e:
            print(f"Skipping row with bad date: {e}")
            continue
        messages = parse_conversation(row["Conversation"])
        if messages:
            by_date[dt.date()].append({
                "datetime": dt,
                "messages": messages,
                "topic": make_topic(messages),
            })

    # Sort conversations within each date by time
    for d in by_date:
        by_date[d].sort(key=lambda c: c["datetime"])

    # Write org files
    for file_date, conversations in sorted(by_date.items()):
        filename = f"{file_date.isoformat()}.org"
        content = build_org_file(conversations)
        (output_dir / filename).write_text(content, encoding="utf-8")
        print(f"  {filename}: {len(conversations)} conversation(s)")

    print(f"\nConverted {len(rows)} conversations into {len(by_date)} org files")


if __name__ == "__main__":
    main()
