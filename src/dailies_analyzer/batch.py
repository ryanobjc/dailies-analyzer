"""Batch API extraction using Anthropic's Message Batches."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from rich.console import Console
from rich.progress import Progress

from .db import Database
from .extractor import EXTRACTION_PROMPT, get_conversation_text, parse_json_response
from .models import ConversationSummary, Insight
from .summarizer import SUMMARY_PROMPT

console = Console()

BATCH_STATE_FILE = Path.home() / ".dailies-analyzer" / "batch_state.json"
SUMMARY_BATCH_STATE_FILE = Path.home() / ".dailies-analyzer" / "summary_batch_state.json"


def prepare_batch_requests(db: Database) -> list[dict]:
    """Prepare batch request entries for unextracted conversations."""
    conversations = db.get_unextracted_conversations()
    requests = []

    for conv in conversations:
        conv_id = conv["id"]
        conv_text = get_conversation_text(db, conv_id)
        topic = conv.get("topic")
        date = conv.get("date")

        # Skip very short conversations
        if len(conv_text) < 200:
            continue

        # Truncate very long conversations
        if len(conv_text) > 50000:
            conv_text = conv_text[:50000] + "\n[TRUNCATED]"

        prompt = EXTRACTION_PROMPT.format(
            conversation=conv_text,
            topic=topic or "Unknown",
            date=date or "Unknown",
        )

        requests.append({
            "custom_id": f"conv_{conv_id}",
            "params": {
                #"model": "claude-sonnet-4-20250514",
                "model": "claude-opus-4-5-20251101",
                "max_tokens": 8196,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    return requests


def submit_batch(db: Database) -> str | None:
    """Submit a batch extraction job. Returns batch ID."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return None

    requests = prepare_batch_requests(db)

    if not requests:
        console.print("[green]No conversations to extract.[/green]")
        return None

    console.print(f"[cyan]Preparing batch of {len(requests)} conversations...[/cyan]")

    # Create JSONL content
    jsonl_lines = [json.dumps(req) for req in requests]
    jsonl_content = "\n".join(jsonl_lines)

    client = Anthropic(api_key=api_key)

    # Submit batch
    console.print("[cyan]Submitting batch to Anthropic...[/cyan]")

    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": req["custom_id"],
                "params": req["params"],
            }
            for req in requests
        ]
    )

    batch_id = batch.id
    console.print(f"[green]Batch submitted![/green] ID: {batch_id}")
    console.print(f"Status: {batch.processing_status}")

    # Save batch state
    state = {
        "batch_id": batch_id,
        "submitted_at": datetime.now().isoformat(),
        "request_count": len(requests),
        "conversation_ids": [int(r["custom_id"].split("_")[1]) for r in requests],
    }
    BATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATE_FILE.write_text(json.dumps(state, indent=2))

    console.print(f"\n[dim]Check status with: dailies batch-status[/dim]")
    console.print(f"[dim]Process results with: dailies batch-results[/dim]")

    return batch_id


def check_batch_status() -> dict | None:
    """Check status of the current batch job."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return None

    if not BATCH_STATE_FILE.exists():
        console.print("[yellow]No batch job found. Run 'dailies batch-extract' first.[/yellow]")
        return None

    state = json.loads(BATCH_STATE_FILE.read_text())
    batch_id = state["batch_id"]

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    console.print(f"[bold]Batch ID:[/bold] {batch_id}")
    console.print(f"[bold]Status:[/bold] {batch.processing_status}")
    console.print(f"[bold]Submitted:[/bold] {state['submitted_at']}")
    console.print(f"[bold]Requests:[/bold] {state['request_count']}")

    if hasattr(batch, "request_counts"):
        counts = batch.request_counts
        console.print(f"\n[bold]Progress:[/bold]")
        console.print(f"  Succeeded: {counts.succeeded}")
        console.print(f"  Errored: {counts.errored}")
        console.print(f"  Canceled: {counts.canceled}")
        console.print(f"  Expired: {counts.expired}")
        console.print(f"  Processing: {counts.processing}")

    return {
        "batch_id": batch_id,
        "status": batch.processing_status,
        "batch": batch,
    }


def process_batch_results(db: Database) -> int:
    """Process results from a completed batch job. Returns insight count."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return 0

    if not BATCH_STATE_FILE.exists():
        console.print("[yellow]No batch job found.[/yellow]")
        return 0

    state = json.loads(BATCH_STATE_FILE.read_text())
    batch_id = state["batch_id"]

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        console.print(f"[yellow]Batch not complete. Status: {batch.processing_status}[/yellow]")
        return 0

    console.print(f"[cyan]Processing batch results...[/cyan]")

    total_insights = 0
    errors = 0

    # Stream results
    with Progress() as progress:
        task = progress.add_task("Processing results...", total=state["request_count"])

        for result in client.messages.batches.results(batch_id):
            progress.advance(task)

            if result.result.type == "error":
                errors += 1
                continue

            if result.result.type != "succeeded":
                continue

            # Extract conversation ID from custom_id
            conv_id = int(result.custom_id.split("_")[1])

            # Get the response content
            message = result.result.message
            if not message.content:
                continue

            content = message.content[0].text

            try:
                insights_data = parse_json_response(content)
            except json.JSONDecodeError:
                errors += 1
                continue

            # Get first message ID for this conversation
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
                except (KeyError, ValueError):
                    pass

    console.print(f"[green]Processed {total_insights} insights[/green]")
    if errors:
        console.print(f"[yellow]{errors} errors encountered[/yellow]")

    # Clean up state file
    BATCH_STATE_FILE.unlink()
    console.print("[dim]Batch state cleared.[/dim]")

    return total_insights


# --- Summary Batch Functions ---


def prepare_summary_batch_requests(db: Database, min_messages: int = 4) -> list[dict]:
    """Prepare batch request entries for unsummarized conversations."""
    conversations = db.get_unsummarized_conversations(min_messages)
    requests = []

    for conv in conversations:
        conv_id = conv["id"]
        conv_text = get_conversation_text(db, conv_id)
        topic = conv.get("topic")
        date = conv.get("date")
        message_count = conv.get("message_count", 0)

        # Skip very short conversations
        if len(conv_text) < 200:
            continue

        # Truncate very long conversations
        if len(conv_text) > 80000:
            conv_text = conv_text[:80000] + "\n[TRUNCATED]"

        prompt = SUMMARY_PROMPT.format(
            conversation=conv_text,
            topic=topic or "Unknown",
            date=date or "Unknown",
            message_count=message_count,
        )

        requests.append({
            "custom_id": f"summary_{conv_id}",
            "params": {
                "model": "claude-opus-4-5-20251101",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    return requests


def submit_summary_batch(db: Database, min_messages: int = 4) -> str | None:
    """Submit a batch summarization job. Returns batch ID."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return None

    requests = prepare_summary_batch_requests(db, min_messages)

    if not requests:
        console.print("[green]No conversations to summarize.[/green]")
        return None

    console.print(f"[cyan]Preparing summary batch of {len(requests)} conversations...[/cyan]")

    client = Anthropic(api_key=api_key)

    # Submit batch
    console.print("[cyan]Submitting batch to Anthropic...[/cyan]")

    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": req["custom_id"],
                "params": req["params"],
            }
            for req in requests
        ]
    )

    batch_id = batch.id
    console.print(f"[green]Batch submitted![/green] ID: {batch_id}")
    console.print(f"Status: {batch.processing_status}")

    # Save batch state
    state = {
        "batch_id": batch_id,
        "submitted_at": datetime.now().isoformat(),
        "request_count": len(requests),
        "conversation_ids": [int(r["custom_id"].split("_")[1]) for r in requests],
    }
    SUMMARY_BATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_BATCH_STATE_FILE.write_text(json.dumps(state, indent=2))

    console.print(f"\n[dim]Check status with: dailies batch-summary-status[/dim]")
    console.print(f"[dim]Process results with: dailies batch-summary-results[/dim]")

    return batch_id


def check_summary_batch_status() -> dict | None:
    """Check status of the current summary batch job."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return None

    if not SUMMARY_BATCH_STATE_FILE.exists():
        console.print("[yellow]No summary batch job found. Run 'dailies batch-summarize' first.[/yellow]")
        return None

    state = json.loads(SUMMARY_BATCH_STATE_FILE.read_text())
    batch_id = state["batch_id"]

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    console.print(f"[bold]Batch ID:[/bold] {batch_id}")
    console.print(f"[bold]Status:[/bold] {batch.processing_status}")
    console.print(f"[bold]Submitted:[/bold] {state['submitted_at']}")
    console.print(f"[bold]Requests:[/bold] {state['request_count']}")

    if hasattr(batch, "request_counts"):
        counts = batch.request_counts
        console.print(f"\n[bold]Progress:[/bold]")
        console.print(f"  Succeeded: {counts.succeeded}")
        console.print(f"  Errored: {counts.errored}")
        console.print(f"  Canceled: {counts.canceled}")
        console.print(f"  Expired: {counts.expired}")
        console.print(f"  Processing: {counts.processing}")

    return {
        "batch_id": batch_id,
        "status": batch.processing_status,
        "batch": batch,
    }


def process_summary_batch_results(db: Database) -> int:
    """Process results from a completed summary batch job. Returns summary count."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        return 0

    if not SUMMARY_BATCH_STATE_FILE.exists():
        console.print("[yellow]No summary batch job found.[/yellow]")
        return 0

    state = json.loads(SUMMARY_BATCH_STATE_FILE.read_text())
    batch_id = state["batch_id"]

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        console.print(f"[yellow]Batch not complete. Status: {batch.processing_status}[/yellow]")
        return 0

    console.print(f"[cyan]Processing summary batch results...[/cyan]")

    total_summaries = 0
    errors = 0

    # Stream results
    with Progress() as progress:
        task = progress.add_task("Processing results...", total=state["request_count"])

        for result in client.messages.batches.results(batch_id):
            progress.advance(task)

            if result.result.type == "error":
                errors += 1
                continue

            if result.result.type != "succeeded":
                continue

            # Extract conversation ID from custom_id
            conv_id = int(result.custom_id.split("_")[1])

            # Get the response content
            message = result.result.message
            if not message.content:
                continue

            content = message.content[0].text

            try:
                # Parse JSON object
                content = content.strip()
                if "```" in content:
                    parts = content.split("```")
                    if len(parts) >= 2:
                        content = parts[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()

                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    content = content[start : end + 1]

                summary_data = json.loads(content)
            except json.JSONDecodeError:
                errors += 1
                continue

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
            except (KeyError, ValueError):
                errors += 1

    console.print(f"[green]Processed {total_summaries} summaries[/green]")
    if errors:
        console.print(f"[yellow]{errors} errors encountered[/yellow]")

    # Clean up state file
    SUMMARY_BATCH_STATE_FILE.unlink()
    console.print("[dim]Summary batch state cleared.[/dim]")

    return total_summaries
