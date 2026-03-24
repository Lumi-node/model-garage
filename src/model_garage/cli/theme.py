"""Model Garage CLI theme - retro mechanic aesthetic."""

from rich.console import Console
from rich.theme import Theme
from rich.table import Table

GARAGE_THEME = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "highlight": "bold magenta",
    "dim": "dim white",
    "part.attention": "bold cyan",
    "part.ffn": "bold green",
    "part.embedding": "bold yellow",
    "part.norm": "dim cyan",
    "part.head": "bold magenta",
    "model": "bold white",
    "layer": "blue",
})

console = Console(theme=GARAGE_THEME)

BANNER = r"""
[cyan]
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
[dim]  Open the hood on neural networks. v{version}[/dim]
"""


def print_banner(version: str = "0.1.0"):
    """Print the Model Garage banner."""
    console.print(BANNER.format(version=version))


def print_model_card(info: dict):
    """Print a model info card in retro style."""
    table = Table(
        title=f"[model]{info.get('model_id', 'Unknown')}[/model]",
        border_style="cyan",
        title_style="bold cyan",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")

    for key, value in info.items():
        if key == "model_id":
            continue
        display_key = key.replace("_", " ").title()
        if isinstance(value, int) and value > 1000:
            display_value = f"{value:,}"
        else:
            display_value = str(value)
        table.add_row(display_key, display_value)

    console.print(table)


def print_parts_table(parts: dict, title: str = "Components"):
    """Print a table of model parts."""
    table = Table(title=title, border_style="cyan")
    table.add_column("Part", style="bold")
    table.add_column("Type", style="part.attention")
    table.add_column("Layer", style="layer")
    table.add_column("Dims", style="dim")

    type_styles = {
        "attention": "part.attention",
        "ffn": "part.ffn",
        "embedding": "part.embedding",
        "layer_norm": "part.norm",
        "output_head": "part.head",
        "full_layer": "bold white",
    }

    for name, part in parts.items():
        part_type = part.part_type.value if hasattr(part.part_type, "value") else str(part.part_type)
        style = type_styles.get(part_type, "white")
        layer = str(part.layer_idx) if part.layer_idx is not None else "-"
        dims = f"{part.input_dim} -> {part.output_dim}"
        table.add_row(name, f"[{style}]{part_type}[/{style}]", layer, dims)

    console.print(table)
