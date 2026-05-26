"""
Regulated pipeline demo: hard gates for clinical/financial use cases.

The key differentiator: soft-ranking is NOT enough for regulated domains.
A stale FDA guideline must be BLOCKED, not just ranked lower.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from ku_weaviate import KUWeaviateClient

load_dotenv()
console = Console()

REGULATED_CONFIG = {
    "clinical_nlp": {
        "decay_threshold": 0.20,
        "velocity_label": "moderate",
        "description": "FDA guidelines, clinical trial data",
    },
    "financial_disclosure": {
        "decay_threshold": 0.35,
        "velocity_label": "fast",
        "description": "SEC filings, compliance documents",
    },
    "general_rag": {
        "decay_threshold": 0.60,
        "velocity_label": "moderate",
        "description": "General purpose RAG pipeline",
    },
}


async def main():
    query = "transformer architecture self-attention mechanisms"

    console.print(f"\n[bold cyan]Cross-Domain Compliance Query:[/] {query}")
    console.print("=" * 70)

    for domain, config in REGULATED_CONFIG.items():
        console.print(
            f"\n[bold magenta]▶ {domain.upper()}[/] — {config['description']}"
        )
        console.print(
            f"  Decay threshold: [bold red]{config['decay_threshold']}[/] | "
            f"Velocity: {config['velocity_label']}"
        )

        client = KUWeaviateClient(
            ku_api_key=os.getenv("KU_API_KEY"),
            weaviate_url=os.getenv("WEAVIATE_URL"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            decay_threshold=config["decay_threshold"],
        )

        try:
            result = await client.query(
                query_text=query,
                limit=5,
                use_governance=True,
                velocity_label=config["velocity_label"],
            )

            table = Table(show_header=True, header_style="bold")
            table.add_column("Platform", style="cyan", width=14)
            table.add_column("Title", width=38)
            table.add_column("Decay", justify="right", width=6)
            table.add_column("Gate", width=12)

            for doc in result.passed_documents:
                table.add_row(
                    doc.get("platform", ""),
                    (doc.get("title", "") or "")[:36],
                    f"{doc.get('decay_score', 0):.2f}",
                    "[green]✓ PASSED[/]",
                )
            for doc in result.blocked_documents:
                table.add_row(
                    doc.get("platform", ""),
                    (doc.get("title", "") or "")[:36],
                    f"{doc.get('decay_score', 0):.2f}",
                    "[red]✗ BLOCKED[/]",
                )

            console.print(table)

            if result.governance_report:
                r = result.governance_report
                console.print(
                    f"  [dim]→ {r.passed}/{r.total_checked} passed "
                    f"({r.block_rate:.0%} block rate)[/]"
                )

        finally:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())