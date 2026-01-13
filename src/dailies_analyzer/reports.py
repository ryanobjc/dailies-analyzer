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
    tag: str | None = None,
    limit: int = 20,
    bottom: bool = False,
):
    """Print extracted insights."""
    import json

    insights = db.get_insights_filtered(
        tag=tag, category=category, ascending=bottom, limit=limit
    )

    if not insights:
        console.print("[yellow]No insights found matching filters.[/yellow]")
        return

    # Build title
    filters = []
    if category:
        filters.append(category)
    if tag:
        filters.append(f"tag:{tag}")
    title_suffix = f" ({', '.join(filters)})" if filters else ""
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
    conv_id = insight.get('conversation_id')
    console.print(f"  Conversation: [dim]#{conv_id}[/dim] {insight['conversation_topic'] or 'Unknown'}")
    console.print(f"  Date: {insight['conversation_date'] or 'Unknown'}")
    console.print(f"  Model: {insight['conversation_model'] or 'Unknown'}")
    console.print(f"  File: {insight['conversation_file'] or 'Unknown'}")
    if conv_id:
        console.print(f"\n[dim]View full conversation: dailies conversation {conv_id}[/dim]")

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


def print_tags(db: Database, limit: int = 50):
    """Print tags sorted by usage count."""
    tags = db.get_tag_counts(limit)

    if not tags:
        console.print("[yellow]No tags found. Run extraction first.[/yellow]")
        return

    total_tags = len(db.get_tag_counts())

    table = Table(title=f"Top {min(limit, len(tags))} Tags (of {total_tags} unique)")
    table.add_column("Tag", style="cyan")
    table.add_column("Count", style="green", justify="right")

    for tag, count in tags:
        table.add_row(tag, str(count))

    console.print(table)


def print_insights_by_tag(
    db: Database,
    tag: str,
    limit: int = 20,
    bottom: bool = False,
):
    """Print insights filtered by tag."""
    import json

    insights = db.get_insights_by_tag(tag, ascending=bottom, limit=limit)

    if not insights:
        console.print(f"[yellow]No insights found with tag '{tag}'[/yellow]")
        return

    title_order = " - Lowest Confidence" if bottom else ""
    table = Table(title=f"Insights tagged '{tag}'{title_order}")
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


def print_deep_conversations(db: Database, limit: int = 20):
    """Print conversations ranked by depth."""
    convs = db.get_deep_conversations(limit)

    if not convs:
        console.print("[yellow]No conversations found.[/yellow]")
        return

    table = Table(title=f"Deepest Conversations (by message count)")
    table.add_column("ID", style="dim", width=5, justify="right")
    table.add_column("Topic", style="cyan", width=35)
    table.add_column("Date", width=10)
    table.add_column("Msgs", justify="right", width=5)
    table.add_column("You", justify="right", width=5)
    table.add_column("AI", justify="right", width=5)
    table.add_column("Tokens", justify="right", width=8)

    for conv in convs:
        table.add_row(
            str(conv["id"]),
            (conv["topic"] or "Unknown")[:35],
            str(conv["date"]) if conv["date"] else "",
            str(conv["message_count"]),
            str(conv["user_messages"]),
            str(conv["assistant_messages"]),
            f"{conv['total_tokens']:,}",
        )

    console.print(table)


def print_conversation_detail(db: Database, conversation_id: int):
    """Print detailed view of a conversation."""
    import json
    from rich.panel import Panel

    conv = db.get_conversation_by_id(conversation_id)

    if not conv:
        console.print(f"[red]Conversation #{conversation_id} not found[/red]")
        return

    # Header
    console.print()
    console.print(Panel(
        f"[bold cyan]{conv['topic'] or 'Untitled'}[/bold cyan]",
        subtitle=f"#{conv['id']} | {conv['message_count']} messages | {conv['total_tokens']:,} tokens",
    ))

    # Metadata
    console.print(f"\n[bold]Metadata:[/bold]")
    console.print(f"  Date: {conv['date'] or 'Unknown'}")
    console.print(f"  Model: {conv['model'] or 'Unknown'}")
    console.print(f"  File: {conv['file_path'] or 'Unknown'}")
    console.print(f"  Your messages: {conv['user_messages']}")
    console.print(f"  AI messages: {conv['assistant_messages']}")

    # Summary (if available)
    summary = db.get_summary(conversation_id)
    if summary:
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  {summary['summary']}")
        topics = json.loads(summary['key_topics']) if summary['key_topics'] else []
        if topics:
            console.print(f"  Topics: {', '.join(topics)}")
        console.print(f"  Sentiment: {summary['sentiment']} | Outcome: {summary['outcome']}")

    # Get related insights
    if db.has_messages_table():
        insights = db.conn.execute(
            """
            SELECT i.* FROM insights i
            JOIN messages m ON i.message_id = m.id
            WHERE m.conversation_id = ?
            ORDER BY i.confidence DESC
            """,
            (conversation_id,),
        ).fetchall()
    else:
        # Export DB: insights linked directly to conversation_id
        insights = db.conn.execute(
            """
            SELECT * FROM insights
            WHERE conversation_id = ?
            ORDER BY confidence DESC
            """,
            (conversation_id,),
        ).fetchall()

    if insights:
        console.print(f"\n[bold]Extracted Insights ({len(insights)}):[/bold]")
        category_colors = {
            "wisdom": "yellow",
            "programming_tip": "cyan",
            "product_idea": "green",
            "question": "magenta",
        }
        for ins in insights[:10]:
            ins = dict(ins)
            cat = ins['category']
            color = category_colors.get(cat, "white")
            console.print(f"  [dim]#{ins['id']}[/dim] [{color}]{cat}[/{color}] {ins['title']} ({ins['confidence']:.0%})")

    # Show conversation preview
    messages = db.get_conversation_messages(conversation_id)
    if messages:
        console.print(f"\n[bold]Conversation Preview:[/bold]")

        for i, msg in enumerate(messages[:6]):
            role_style = "green" if msg["role"] == "user" else "blue"
            role_label = "YOU" if msg["role"] == "user" else "AI"
            content = msg["content"][:200]
            if len(msg["content"]) > 200:
                content += "..."
            console.print(f"\n[{role_style}][{role_label}][/{role_style}]")
            console.print(f"  {content}")

        if len(messages) > 6:
            console.print(f"\n[dim]... {len(messages) - 6} more messages ...[/dim]")
    else:
        console.print(f"\n[dim]Message content not available in this database.[/dim]")


def print_summary_stats(db: Database):
    """Print summary statistics overview."""
    stats = db.get_summary_stats()

    if stats["total"] == 0:
        console.print("[yellow]No summaries found. Run 'dailies batch-summarize' first.[/yellow]")
        return

    console.print(f"\n[bold]Summary Statistics[/bold] ({stats['total']} conversations summarized)\n")

    # Sentiment distribution
    if stats["sentiments"]:
        table = Table(title="By Sentiment")
        table.add_column("Sentiment", style="cyan")
        table.add_column("Count", style="green", justify="right")

        sentiment_colors = {
            "technical": "blue",
            "exploratory": "magenta",
            "debugging": "red",
            "learning": "green",
            "planning": "yellow",
            "creative": "cyan",
            "frustrated": "red",
            "collaborative": "green",
        }

        for sentiment, count in stats["sentiments"]:
            color = sentiment_colors.get(sentiment, "white")
            table.add_row(f"[{color}]{sentiment}[/{color}]", str(count))

        console.print(table)

    console.print()

    # Outcome distribution
    if stats["outcomes"]:
        table = Table(title="By Outcome")
        table.add_column("Outcome", style="cyan")
        table.add_column("Count", style="green", justify="right")

        outcome_colors = {
            "resolved": "green",
            "learning": "blue",
            "decision_made": "yellow",
            "idea_generated": "magenta",
            "ongoing": "cyan",
            "abandoned": "red",
        }

        for outcome, count in stats["outcomes"]:
            color = outcome_colors.get(outcome, "white")
            table.add_row(f"[{color}]{outcome}[/{color}]", str(count))

        console.print(table)


def print_summaries(
    db: Database,
    sentiment: str | None = None,
    outcome: str | None = None,
    limit: int = 20,
):
    """Print summaries with optional filters."""
    import json

    summaries = db.get_summaries_filtered(sentiment=sentiment, outcome=outcome, limit=limit)

    if not summaries:
        console.print("[yellow]No summaries found matching filters.[/yellow]")
        return

    # Build title
    filters = []
    if sentiment:
        filters.append(f"sentiment:{sentiment}")
    if outcome:
        filters.append(f"outcome:{outcome}")
    title_suffix = f" ({', '.join(filters)})" if filters else ""

    table = Table(title=f"Conversation Summaries{title_suffix}")
    table.add_column("ID", style="dim", width=5, justify="right")
    table.add_column("Date", width=10)
    table.add_column("Topic", style="cyan", width=30)
    table.add_column("Sentiment", width=12)
    table.add_column("Outcome", width=14)
    table.add_column("Msgs", justify="right", width=4)

    sentiment_colors = {
        "technical": "blue",
        "exploratory": "magenta",
        "debugging": "red",
        "learning": "green",
        "planning": "yellow",
        "creative": "cyan",
        "frustrated": "red",
        "collaborative": "green",
    }

    outcome_colors = {
        "resolved": "green",
        "learning": "blue",
        "decision_made": "yellow",
        "idea_generated": "magenta",
        "ongoing": "cyan",
        "abandoned": "red",
    }

    for s in summaries:
        sent = s["sentiment"] or "unknown"
        out = s["outcome"] or "unknown"
        sent_color = sentiment_colors.get(sent, "white")
        out_color = outcome_colors.get(out, "white")

        table.add_row(
            str(s["conversation_id"]),
            str(s["date"]) if s["date"] else "",
            (s["topic"] or "Untitled")[:30],
            f"[{sent_color}]{sent}[/{sent_color}]",
            f"[{out_color}]{out}[/{out_color}]",
            str(s["message_count"]) if s["message_count"] else "",
        )

    console.print(table)

    # Show details for first few
    console.print(f"\n[bold]Summary Details:[/bold]\n")
    for s in summaries[:5]:
        console.print(f"[dim]#{s['conversation_id']}[/dim] [cyan]{s['topic'] or 'Untitled'}[/cyan]")
        console.print(f"  {s['summary']}")
        topics = json.loads(s['key_topics']) if s['key_topics'] else []
        if topics:
            console.print(f"  [dim]Topics: {', '.join(topics)}[/dim]")
        console.print()
