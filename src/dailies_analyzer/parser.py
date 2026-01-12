"""Parser for org-mode files with gptel annotations."""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .models import Conversation, Message


@dataclass
class GptelProperties:
    """Properties extracted from gptel annotations."""

    model: str | None = None
    backend: str | None = None
    system: str | None = None
    bounds: list[tuple[int, int]] | None = None
    topic: str | None = None


def parse_gptel_bounds(bounds_str: str) -> list[tuple[int, int]]:
    """Parse GPTEL_BOUNDS property into list of (start, end) tuples.

    Handles both formats:
    - Old: ((1807 . 3547))
    - New: ((response (1116 2260) (2648 3860) ...))
    """
    if not bounds_str:
        return []

    # New format: ((response (start end) (start end) ...))
    if "response" in bounds_str:
        # Extract all (start end) pairs after 'response'
        matches = re.findall(r"\((\d+)\s+(\d+)\)", bounds_str)
        return [(int(start), int(end)) for start, end in matches]

    # Old format: ((start . end) (start . end) ...)
    matches = re.findall(r"\((\d+)\s*\.\s*(\d+)\)", bounds_str)
    return [(int(start), int(end)) for start, end in matches]


def extract_properties_block(content: str, start_pos: int = 0) -> tuple[GptelProperties, int]:
    """Extract GPTEL properties from a :PROPERTIES: block.

    Returns properties and the position after the block.
    """
    props = GptelProperties()

    # Find :PROPERTIES: block
    props_match = re.search(r":PROPERTIES:\s*\n(.*?):END:", content[start_pos:], re.DOTALL)
    if not props_match:
        return props, start_pos

    props_text = props_match.group(1)
    end_pos = start_pos + props_match.end()

    # Extract individual properties
    for line in props_text.split("\n"):
        line = line.strip()
        if line.startswith(":GPTEL_MODEL:"):
            props.model = line.split(":", 2)[2].strip()
        elif line.startswith(":GPTEL_BACKEND:"):
            props.backend = line.split(":", 2)[2].strip()
        elif line.startswith(":GPTEL_SYSTEM:"):
            props.system = line.split(":", 2)[2].strip()
        elif line.startswith(":GPTEL_BOUNDS:"):
            bounds_str = line.split(":", 2)[2].strip()
            props.bounds = parse_gptel_bounds(bounds_str)
        elif line.startswith(":GPTEL_TOPIC:"):
            props.topic = line.split(":", 2)[2].strip()

    return props, end_pos


def strip_org_formatting(text: str) -> str:
    """Strip org-mode formatting to plain text."""
    # Remove org-mode markup
    text = re.sub(r"^\*+\s+", "", text, flags=re.MULTILINE)  # Headlines
    text = re.sub(r"#\+begin_src.*?#\+end_src", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"#\+begin_quote.*?#\+end_quote", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"#\+begin_example.*?#\+end_example", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"#\+RESULTS:.*?\n", "", text)
    text = re.sub(r"\[\[.*?\]\[?(.*?)\]?\]", r"\1", text)  # Links
    text = re.sub(r"^:PROPERTIES:.*?:END:\s*", "", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^#\+.*$", "", text, flags=re.MULTILINE)  # Other directives
    text = re.sub(r"@(user|assistant)\s*", "", text)  # gptel markers

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_messages_from_bounds(
    content: str, bounds: list[tuple[int, int]]
) -> list[Message]:
    """Extract user and assistant messages based on GPTEL_BOUNDS.

    Bounds indicate assistant response regions. Everything else is user input.
    """
    if not bounds:
        return []

    messages = []
    sorted_bounds = sorted(bounds, key=lambda x: x[0])

    # Start from beginning of content
    current_pos = 0

    for start, end in sorted_bounds:
        # User content before this assistant response
        if start > current_pos:
            user_content = content[current_pos:start]
            user_text = strip_org_formatting(user_content)
            if user_text:
                messages.append(
                    Message(
                        role="user",
                        content=user_text,
                        char_start=current_pos,
                        char_end=start,
                    )
                )

        # Assistant response
        assistant_content = content[start:end]
        assistant_text = strip_org_formatting(assistant_content)
        if assistant_text:
            messages.append(
                Message(
                    role="assistant",
                    content=assistant_text,
                    char_start=start,
                    char_end=end,
                )
            )

        current_pos = end

    # Any remaining content after last assistant response is user input
    if current_pos < len(content):
        user_content = content[current_pos:]
        user_text = strip_org_formatting(user_content)
        if user_text:
            messages.append(
                Message(
                    role="user",
                    content=user_text,
                    char_start=current_pos,
                    char_end=len(content),
                )
            )

    return messages


def parse_date_from_filename(filepath: Path) -> date | None:
    """Extract date from org-roam daily filename (YYYY-MM-DD.org)."""
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})\.org$", filepath.name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


@dataclass
class Section:
    """A section of an org file (top-level headline)."""

    title: str
    start_pos: int
    end_pos: int
    topic: str | None = None


def find_top_level_sections(content: str) -> list[Section]:
    """Find all top-level headlines and their positions."""
    sections = []

    # Find all top-level headlines (single asterisk at start of line)
    for match in re.finditer(r"^(\* .+)$", content, re.MULTILINE):
        title = match.group(1)[2:].strip()  # Remove "* " prefix
        start_pos = match.start()
        sections.append(Section(title=title, start_pos=start_pos, end_pos=len(content)))

    # Set end positions (each section ends where the next begins)
    for i in range(len(sections) - 1):
        sections[i].end_pos = sections[i + 1].start_pos

    # Extract GPTEL_TOPIC from each section's properties
    for section in sections:
        section_content = content[section.start_pos : section.end_pos]
        props, _ = extract_properties_block(section_content)
        section.topic = props.topic

    return sections


def filter_bounds_for_section(
    bounds: list[tuple[int, int]], section_start: int, section_end: int
) -> list[tuple[int, int]]:
    """Filter bounds to only include those within a section's range."""
    return [
        (start, end)
        for start, end in bounds
        if start >= section_start and end <= section_end
    ]


def parse_org_file(filepath: Path) -> list[Conversation]:
    """Parse an org file and extract conversations.

    Returns a list of conversations - one per top-level headline section.
    Each section is treated as a separate conversation context.
    """
    content = filepath.read_text(encoding="utf-8")
    file_date = parse_date_from_filename(filepath)

    # Get file-level properties (contains ALL bounds for the file)
    file_props, file_props_end = extract_properties_block(content)

    if not file_props.bounds:
        # No gptel content in this file
        return []

    # Find all top-level sections
    sections = find_top_level_sections(content)

    conversations = []

    if not sections:
        # No headlines - treat whole file as one conversation
        messages = extract_messages_from_bounds(content, file_props.bounds)
        if messages:
            conversations.append(
                Conversation(
                    file_path=str(filepath),
                    date=file_date,
                    topic=file_props.topic,
                    model=file_props.model,
                    system_prompt=file_props.system,
                    messages=messages,
                )
            )
    else:
        # Split by sections
        for section in sections:
            # Filter bounds to only those within this section
            section_bounds = filter_bounds_for_section(
                file_props.bounds, section.start_pos, section.end_pos
            )

            if not section_bounds:
                continue

            # Extract messages for this section
            section_content = content[section.start_pos : section.end_pos]
            # Adjust bounds to be relative to section start
            adjusted_bounds = [
                (start - section.start_pos, end - section.start_pos)
                for start, end in section_bounds
            ]
            messages = extract_messages_from_bounds(section_content, adjusted_bounds)

            if messages:
                conversations.append(
                    Conversation(
                        file_path=str(filepath),
                        date=file_date,
                        topic=section.topic or section.title,
                        model=file_props.model,
                        system_prompt=file_props.system,
                        messages=messages,
                    )
                )

    return conversations


def parse_directory(directory: Path) -> list[Conversation]:
    """Parse all org files in a directory."""
    conversations = []

    for filepath in sorted(directory.glob("*.org")):
        try:
            file_conversations = parse_org_file(filepath)
            conversations.extend(file_conversations)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

    return conversations
