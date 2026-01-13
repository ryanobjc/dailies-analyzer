"""CLI entry point for dailies analyzer."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from .db import Database
from .parser import parse_directory
from .reports import (
    print_conversation_detail,
    print_deep_conversations,
    print_insight_detail,
    print_insights,
    print_model_distribution,
    print_random_insight,
    print_summary,
    print_tags,
    print_top_days,
    print_topic_distribution,
)
from .stats import compute_and_store_stats, count_tokens

console = Console()

DEFAULT_DB = Path.home() / ".dailies-analyzer" / "dailies.db"


@click.group()
@click.option(
    "--db",
    type=click.Path(),
    default=str(DEFAULT_DB),
    help="Path to SQLite database",
)
@click.pass_context
def cli(ctx, db):
    """Analyze your org-mode/gptel chat history."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--clear", is_flag=True, help="Clear existing data before ingesting")
@click.pass_context
def ingest(ctx, directory, clear):
    """Ingest org-mode files from a directory."""
    db_path = ctx.obj["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    directory = Path(directory)
    console.print(f"[cyan]Parsing files from {directory}...[/cyan]")

    # Parse all files
    conversations = parse_directory(directory)
    console.print(f"Found [green]{len(conversations)}[/green] conversations")

    if not conversations:
        console.print("[yellow]No gptel conversations found in directory.[/yellow]")
        return

    # Count tokens for all messages
    console.print("[cyan]Counting tokens...[/cyan]")
    with Progress() as progress:
        task = progress.add_task("Processing messages...", total=sum(len(c.messages) for c in conversations))
        for conv in conversations:
            for msg in conv.messages:
                msg.token_count = count_tokens(msg.content)
                progress.advance(task)

    # Store in database
    console.print(f"[cyan]Storing in {db_path}...[/cyan]")
    with Database(db_path) as db:
        db.init_schema()
        if clear:
            db.clear_all()
            console.print("[yellow]Cleared existing data[/yellow]")

        db.insert_conversations(conversations)

        # Compute stats
        console.print("[cyan]Computing statistics...[/cyan]")
        compute_and_store_stats(db)

        console.print(f"[green]Done![/green] {db.get_conversation_count()} conversations, {db.get_message_count()} messages")


@cli.command()
@click.option("--top", default=10, help="Number of top days to show")
@click.option("--models", is_flag=True, help="Show model distribution")
@click.option("--topics", is_flag=True, help="Show topic distribution")
@click.pass_context
def stats(ctx, top, models, topics):
    """Show statistics about your chat history."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        console.print("Run 'dailies ingest <directory>' first.")
        return

    with Database(db_path) as db:
        print_summary(db)
        console.print()

        if models:
            print_model_distribution(db)
            console.print()

        if topics:
            print_topic_distribution(db)
            console.print()

        print_top_days(db, top)


@cli.command()
@click.option("--category", type=click.Choice(["wisdom", "product_idea", "programming_tip", "question"]))
@click.option("--tag", help="Filter by tag")
@click.option("--limit", default=20, help="Maximum insights to show")
@click.option("--bottom", is_flag=True, help="Show lowest confidence instead of highest")
@click.pass_context
def insights(ctx, category, tag, limit, bottom):
    """Show extracted insights. Use --category and --tag together to intersect."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    with Database(db_path) as db:
        print_insights(db, category, tag, limit, bottom)


@cli.command()
@click.option("--limit", default=50, help="Maximum tags to show")
@click.pass_context
def tags(ctx, limit):
    """Show top tags by usage count."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    with Database(db_path) as db:
        print_tags(db, limit)


@cli.command()
@click.argument("insight_id", type=int)
@click.pass_context
def insight(ctx, insight_id):
    """Show detailed view of a specific insight by ID."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    with Database(db_path) as db:
        print_insight_detail(db, insight_id)


@cli.command()
@click.option("--category", type=click.Choice(["wisdom", "product_idea", "programming_tip", "question"]))
@click.pass_context
def random(ctx, category):
    """Show a random insight."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    with Database(db_path) as db:
        print_random_insight(db, category)


@cli.command()
@click.option("--limit", default=20, help="Number of conversations to show")
@click.pass_context
def deep(ctx, limit):
    """Show deepest conversations by message count."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    with Database(db_path) as db:
        print_deep_conversations(db, limit)


@cli.command()
@click.argument("conversation_id", type=int)
@click.pass_context
def conversation(ctx, conversation_id):
    """Show detailed view of a specific conversation."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    with Database(db_path) as db:
        print_conversation_detail(db, conversation_id)


@cli.command()
@click.option("--limit", default=10, help="Number of conversations to process")
@click.option("--all", "process_all", is_flag=True, help="Process all unextracted conversations")
@click.pass_context
def extract(ctx, limit, process_all):
    """Extract insights using Claude API."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    # Import here to avoid loading anthropic unless needed
    from .extractor import extract_insights

    with Database(db_path) as db:
        unextracted = db.get_unextracted_conversations()

        if not unextracted:
            console.print("[green]All conversations have been processed![/green]")
            return

        to_process = unextracted if process_all else unextracted[:limit]
        console.print(f"[cyan]Processing {len(to_process)} of {len(unextracted)} unextracted conversations...[/cyan]")

        extract_insights(db, to_process)


@cli.command()
@click.argument("output", type=click.Path())
@click.pass_context
def export(ctx, output):
    """Export database for sharing (without message content)."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    output_path = Path(output)

    with Database(db_path) as db:
        stats = db.export_to_file(output_path)

    console.print(f"[green]Exported to {output_path}[/green]")
    console.print(f"  Conversations: {stats['conversations']}")
    console.print(f"  Daily stats: {stats['daily_stats']}")
    console.print(f"  Insights: {stats['insights']}")
    console.print(f"  Tags: {stats['tags']}")
    console.print(f"\n[dim]Message content excluded for privacy.[/dim]")


@cli.command("batch-extract")
@click.pass_context
def batch_extract(ctx):
    """Submit batch extraction job (50% cheaper, async)."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    from .batch import submit_batch

    with Database(db_path) as db:
        submit_batch(db)


@cli.command("batch-status")
@click.pass_context
def batch_status(ctx):
    """Check status of batch extraction job."""
    from .batch import check_batch_status
    check_batch_status()


@cli.command("batch-results")
@click.pass_context
def batch_results(ctx):
    """Process results from completed batch job."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    from .batch import process_batch_results

    with Database(db_path) as db:
        process_batch_results(db)


@cli.command()
@click.option("--limit", default=10, help="Number of conversations to summarize")
@click.option("--all", "process_all", is_flag=True, help="Summarize all unsummarized conversations")
@click.option("--min-messages", default=4, help="Minimum messages for a conversation to be summarized")
@click.pass_context
def summarize(ctx, limit, process_all, min_messages):
    """Summarize conversations using Claude Opus (sync, slower)."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    from .summarizer import summarize_conversations

    with Database(db_path) as db:
        db.init_schema()  # Ensure new table exists
        unsummarized = db.get_unsummarized_conversations(min_messages)

        if not unsummarized:
            console.print("[green]All conversations have been summarized![/green]")
            return

        to_process = unsummarized if process_all else unsummarized[:limit]
        console.print(f"[cyan]Summarizing {len(to_process)} of {len(unsummarized)} unsummarized conversations...[/cyan]")

        summarize_conversations(db, to_process)


@cli.command("batch-summarize")
@click.option("--min-messages", default=4, help="Minimum messages for a conversation to be summarized")
@click.pass_context
def batch_summarize(ctx, min_messages):
    """Submit batch summarization job (50% cheaper, async)."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    from .batch import submit_summary_batch

    with Database(db_path) as db:
        db.init_schema()  # Ensure new table exists
        submit_summary_batch(db, min_messages)


@cli.command("batch-summary-status")
@click.pass_context
def batch_summary_status(ctx):
    """Check status of batch summarization job."""
    from .batch import check_summary_batch_status
    check_summary_batch_status()


@cli.command("batch-summary-results")
@click.pass_context
def batch_summary_results(ctx):
    """Process results from completed batch summarization job."""
    db_path = ctx.obj["db_path"]

    if not db_path.exists():
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    from .batch import process_summary_batch_results

    with Database(db_path) as db:
        process_summary_batch_results(db)


if __name__ == "__main__":
    cli()
