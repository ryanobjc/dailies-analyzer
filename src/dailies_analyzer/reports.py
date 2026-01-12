"""Report generation for CLI output."""

from rich.console import Console
from rich.table import Table

from .db import Database
from .stats import (
    get_model_distribution,
    get_summary_stats,
    get_top_days,
    get_topic_distribution,
)

console = Console()


def print_summary(db: Database):
    """Print overall summary statistics."""
    stats = get_summary_stats(db)

    if stats["total_days"] == 0:
        console.print("[yellow]No data found. Run 'dailies ingest' first.[/yellow]")
        return

    table = Table(title="Summary Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Date Range", f"{stats['first_date']} to {stats['last_date']}")
    table.add_row("Active Days", str(stats["total_days"]))
    table.add_row("Total Conversations", str(stats["total_conversations"]))
    table.add_row("Total Messages", str(stats["total_messages"]))
    table.add_row("Your Tokens", f"{stats['total_user_tokens']:,}")
    table.add_row("Assistant Tokens", f"{stats['total_assistant_tokens']:,}")
    table.add_row(
        "Total Tokens",
        f"{stats['total_user_tokens'] + stats['total_assistant_tokens']:,}",
    )
    table.add_row("Avg Messages/Day", f"{stats['avg_messages_per_day']:.1f}")

    most_active = stats["most_active_day"]
    table.add_row(
        "Most Active Day",
        f"{most_active.date} ({most_active.total_messages} messages)",
    )

    console.print(table)


def print_top_days(db: Database, limit: int = 10):
    """Print top N most active days."""
    top = get_top_days(db, limit)

    if not top:
        console.print("[yellow]No data found.[/yellow]")
        return

    table = Table(title=f"Top {limit} Most Active Days")
    table.add_column("Date", style="cyan")
    table.add_column("Messages", style="green", justify="right")
    table.add_column("Your Tokens", justify="right")
    table.add_column("Assistant Tokens", justify="right")
    table.add_column("Conversations", justify="right")

    for day in top:
        table.add_row(
            str(day.date),
            str(day.total_messages),
            f"{day.user_tokens:,}",
            f"{day.assistant_tokens:,}",
            str(day.conversation_count),
        )

    console.print(table)


def print_model_distribution(db: Database):
    """Print distribution of usage by model."""
    dist = get_model_distribution(db)

    if not dist:
        console.print("[yellow]No model data found.[/yellow]")
        return

    table = Table(title="Messages by Model")
    table.add_column("Model", style="cyan")
    table.add_column("Messages", style="green", justify="right")

    for model, count in dist.items():
        table.add_row(model, str(count))

    console.print(table)


def print_topic_distribution(db: Database, limit: int = 20):
    """Print distribution of conversations by topic."""
    dist = get_topic_distribution(db)

    if not dist:
        console.print("[yellow]No topic data found.[/yellow]")
        return

    table = Table(title=f"Top {limit} Topics")
    table.add_column("Topic", style="cyan")
    table.add_column("Conversations", style="green", justify="right")

    for topic, count in list(dist.items())[:limit]:
        table.add_row(topic, str(count))

    console.print(table)


def print_insights(
    db: Database,
    category: str | None = None,
    limit: int = 20,
    bottom: bool = False,
):
    """Print extracted insights."""
    import json

    insights = db.get_insights(category, ascending=bottom)[:limit]

    if not insights:
        console.print("[yellow]No insights found. Run 'dailies extract' first.[/yellow]")
        return

    title_suffix = f" ({category})" if category else ""
    title_order = " - Lowest Confidence" if bottom else ""
    table = Table(title=f"Insights{title_suffix}{title_order}")
    table.add_column("ID", style="dim", width=6, justify="right")
    table.add_column("Category", style="magenta", width=15)
    table.add_column("Title", style="cyan", width=40)
    table.add_column("Confidence", justify="right", width=10)

    for insight in insights:
        table.add_row(
            str(insight["id"]),
            insight["category"],
            insight["title"],
            f"{insight['confidence']:.0%}",
        )

    console.print(table)

    # Print details for top insights
    label = "Bottom" if bottom else "Top"
    console.print(f"\n[bold]{label} Insight Details:[/bold]\n")
    for insight in insights[:5]:
        console.print(f"[dim]#{insight['id']}[/dim] [cyan]{insight['title']}[/cyan]")
        console.print(f"  Category: {insight['category']}")
        console.print(f"  {insight['summary']}")
        tags = json.loads(insight["tags"]) if insight["tags"] else []
        if tags:
            console.print(f"  Tags: {', '.join(tags)}")
        console.print()


def print_insight_detail(db: Database, insight_id: int):
    """Print detailed view of a single insight with context."""
    import json
    from rich.panel import Panel
    from rich.markdown import Markdown

    insight = db.get_insight_with_context(insight_id)

    if not insight:
        console.print(f"[red]Insight #{insight_id} not found[/red]")
        return

    # Header
    console.print()
    console.print(Panel(
        f"[bold cyan]{insight['title']}[/bold cyan]",
        subtitle=f"#{insight['id']} | {insight['category']} | {insight['confidence']:.0%} confidence",
    ))

    # Summary
    console.print(f"\n[bold]Summary:[/bold]\n{insight['summary']}\n")

    # Tags
    tags = json.loads(insight["tags"]) if insight["tags"] else []
    if tags:
        console.print(f"[bold]Tags:[/bold] {', '.join(tags)}\n")

    # Context
    console.print("[bold]Context:[/bold]")
    console.print(f"  Topic: {insight['conversation_topic'] or 'Unknown'}")
    console.print(f"  Date: {insight['conversation_date'] or 'Unknown'}")
    console.print(f"  Model: {insight['conversation_model'] or 'Unknown'}")
    console.print(f"  File: {insight['conversation_file'] or 'Unknown'}")

    # Source message (truncated if long)
    if insight["message_content"]:
        console.print(f"\n[bold]Source Message ({insight['message_role']}):[/bold]")
        content = insight["message_content"]
        if len(content) > 1500:
            content = content[:1500] + "\n... [truncated]"
        console.print(Panel(content, border_style="dim"))


def print_random_insight(db: Database, category: str | None = None):
    """Print a random insight."""
    import json

    insight = db.get_random_insight(category)

    if not insight:
        console.print("[yellow]No insights found.[/yellow]")
        return

    print_insight_detail(db, insight["id"])
