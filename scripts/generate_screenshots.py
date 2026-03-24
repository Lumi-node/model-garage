#!/usr/bin/env python3
"""
Generate SVG terminal screenshots for README and docs.

Usage:
    python scripts/generate_screenshots.py

Outputs SVGs to assets/screenshots/
"""

import sys
from pathlib import Path

# Ensure model_garage is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from model_garage.cli.theme import GARAGE_THEME

ASSETS = Path(__file__).parent.parent / "assets" / "screenshots"
ASSETS.mkdir(parents=True, exist_ok=True)


def screenshot_banner():
    """Generate the banner screenshot."""
    console = Console(theme=GARAGE_THEME, record=True, width=72)

    banner = r"""[cyan]
 ╔══════════════════════════════════════════════════════════╗
 ║  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗                ║
 ║  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║                ║
 ║  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║                ║
 ║  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║                ║
 ║  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗           ║
 ║  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝           ║
 ║   ██████╗  █████╗ ██████╗  █████╗  ██████╗ ███████╗     ║
 ║  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝     ║
 ║  ██║  ███╗███████║██████╔╝███████║██║  ███╗█████╗       ║
 ║  ██║   ██║██╔══██║██╔══██╗██╔══██║██║   ██║██╔══╝       ║
 ║  ╚██████╔╝██║  ██║██║  ██║██║  ██║╚██████╔╝███████╗     ║
 ║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝     ║
 ╚══════════════════════════════════════════════════════════╝
[/cyan]
[dim]  Open the hood on neural networks. v0.1.0[/dim]
"""
    console.print(banner)
    svg = console.export_svg(title="Model Garage")
    (ASSETS / "banner.svg").write_text(svg)
    print("  Saved banner.svg")


def screenshot_model_card():
    """Generate a model card screenshot."""
    console = Console(theme=GARAGE_THEME, record=True, width=60)

    console.print("[dim]$ garage open gpt2[/dim]\n")

    table = Table(
        title="[bold cyan]gpt2[/bold cyan]",
        border_style="cyan",
        title_style="bold cyan",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Architecture", "GPT2LMHeadModel")
    table.add_row("Hidden Size", "768")
    table.add_row("Num Layers", "12")
    table.add_row("Num Heads", "12")
    table.add_row("Vocab Size", "50,257")
    table.add_row("Max Position", "1,024")
    table.add_row("Total Params", "124,439,808")
    table.add_row("Device", "cuda")
    table.add_row("Dtype", "torch.float16")

    console.print(table)
    svg = console.export_svg(title="garage open gpt2")
    (ASSETS / "model_card.svg").write_text(svg)
    print("  Saved model_card.svg")


def screenshot_extract():
    """Generate an extract components screenshot."""
    console = Console(theme=GARAGE_THEME, record=True, width=72)

    console.print("[dim]$ garage extract gpt2 --layer 6[/dim]\n")
    console.print("[green]4 components from layer 6[/green]\n")

    table = Table(border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Type", style="bold cyan")
    table.add_column("Params", justify="right", style="bold")
    table.add_column("Dims", style="dim")

    table.add_row("self_attention", "[bold cyan]attention[/bold cyan]", "2,362,368", "768 → 768")
    table.add_row("feed_forward", "[bold green]ffn[/bold green]", "4,722,432", "768 → 768")
    table.add_row("input_norm", "[dim cyan]norm[/dim cyan]", "768", "768 → 768")
    table.add_row("post_attn_norm", "[dim cyan]norm[/dim cyan]", "768", "768 → 768")

    console.print(table)
    svg = console.export_svg(title="garage extract gpt2 --layer 6")
    (ASSETS / "extract.svg").write_text(svg)
    print("  Saved extract.svg")


def screenshot_analyze():
    """Generate an activation analysis screenshot."""
    console = Console(theme=GARAGE_THEME, record=True, width=68)

    console.print('[dim]$ garage analyze gpt2 --prompt "The meaning of life"[/dim]\n')

    table = Table(title="Activation Analysis", border_style="cyan")
    table.add_column("Layer", style="blue")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Sparsity", justify="right")

    data = [
        ("0", "-0.0012", "0.5834", "0.00%"),
        ("1", "0.0089", "1.2341", "0.00%"),
        ("2", "-0.0234", "1.8912", "0.01%"),
        ("3", "0.0156", "2.4567", "0.02%"),
        ("4", "-0.0078", "3.1234", "0.03%"),
        ("5", "0.0345", "3.8901", "0.05%"),
        ("6", "-0.0123", "4.5678", "0.08%"),
        ("7", "0.0567", "5.2345", "0.12%"),
        ("8", "-0.0234", "5.9012", "0.18%"),
        ("9", "0.0789", "6.5679", "0.25%"),
        ("10", "-0.0456", "7.2346", "0.34%"),
        ("11", "0.0912", "7.9013", "0.45%"),
    ]

    for row in data:
        table.add_row(*row)

    console.print(table)
    svg = console.export_svg(title="garage analyze gpt2")
    (ASSETS / "analyze.svg").write_text(svg)
    print("  Saved analyze.svg")


def screenshot_compare():
    """Generate a model comparison screenshot."""
    console = Console(theme=GARAGE_THEME, record=True, width=60)

    console.print("[dim]$ garage compare gpt2 distilgpt2[/dim]\n")

    table = Table(title="Architecture Comparison", border_style="cyan")
    table.add_column("Property", style="dim")
    table.add_column("gpt2", style="cyan", justify="right")
    table.add_column("distilgpt2", style="green", justify="right")

    table.add_row("Families", "gpt2", "gpt2")
    table.add_row("Hidden Dims", "768", "768")
    table.add_row("Num Layers", "12", "6")
    table.add_row("Num Heads", "12", "12")
    table.add_row("Vocab Sizes", "50,257", "50,257")

    console.print(table)
    console.print("\n[green]Same hidden dimension — parts are swappable![/green]")

    svg = console.export_svg(title="garage compare gpt2 distilgpt2")
    (ASSETS / "compare.svg").write_text(svg)
    print("  Saved compare.svg")


if __name__ == "__main__":
    print("Generating screenshots...")
    screenshot_banner()
    screenshot_model_card()
    screenshot_extract()
    screenshot_analyze()
    screenshot_compare()
    print(f"\nDone! Screenshots saved to {ASSETS}/")
