"""
Full two-layer RAG pipeline demonstration.
Shows the before/after: with and without governance.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from ku_weaviate import KUWeaviateClient

load_dotenv()
console = Console()


async def main():
    client = KUWeaviateClient(
        ku_api_key=os.getenv("KU_API_KEY"),
        weaviate_url=os.getenv("WEAVIATE_URL"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        decay_threshold=0.40,
    )

    try:
        query = "how does self-attention work in transformers"

        console.print(f"\n[bold cyan]Query:[/] {query}")
        console.print("[bold]" + "─" * 60)

        # Layer 1 only — Weaviate Boost.decay, no hard gates
        console.print("\n[bold yellow]LAYER 1 ONLY: Weaviate vector search[/]")
        result_l1 = await client.query(
            query, limit=5, use_governance=False
        )
        _print_results(result_l1, console)

        # Both layers — vector search + KU hard gating
        console.print("\n[bold green]BOTH LAYERS: Vector search + KU Governance[/]")
        result_both = await client.query(
            query,
            limit=5,
            use_governance=True,
            velocity_label="moderate",
        )
        _print_results(result_both, console)

        # Summary
        if result_both.governance_report:
            report = result_both.governance_report
            console.print(
                f"\n[bold]Governance summary:[/] "
                f"{report.passed} passed, "
                f"{report.blocked} blocked "
                f"({report.block_rate:.0%} block rate)"
            )
            for r in report.results:
                if not r.passed:
                    console.print(
                        f"  [red]✗[/] [{r.platform}] "
                        f"decay={r.decay_score:.2f} → {r.block_reason}"
                    )

    finally:
        await client.aclose()


def _print_results(result, console):
    if not result.passed_documents and not result.blocked_documents:
        console.print("  [dim]No results — ingest data first with 01_ingest.py[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Platform", style="cyan", width=14)
    table.add_column("Title", width=38)
    table.add_column("Decay", justify="right", width=6)
    table.add_column("Quality", justify="right", width=7)
    table.add_column("Status", width=12)

    for doc in result.passed_documents:
        table.add_row(
            doc.get("platform", ""),
            (doc.get("title", "") or "")[:36],
            f"{doc.get('decay_score', 0):.2f}",
            f"{doc.get('quality_score', 0):.1f}",
            "[green]✓ passed[/]",
        )
    for doc in result.blocked_documents[:3]:
        table.add_row(
            doc.get("platform", ""),
            (doc.get("title", "") or "")[:36],
            f"{doc.get('decay_score', 0):.2f}",
            f"{doc.get('quality_score', 0):.1f}",
            "[red]✗ blocked[/]",
        )

    console.print(table)
    console.print(
        f"  [dim]Candidates: {result.total_candidates} | "
        f"Passed: {result.passed_count} | "
        f"Blocked: {result.blocked_count}[/]"
    )


if __name__ == "__main__":
    asyncio.run(main())