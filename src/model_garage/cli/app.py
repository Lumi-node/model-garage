"""Model Garage CLI - component-level neural network surgery."""

import typer
from typing import Optional

app = typer.Typer(
    name="garage",
    help="Open the hood on neural networks.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    """Model Garage - component-level neural network surgery."""
    if version:
        from model_garage import __version__
        from model_garage.cli.theme import console

        console.print(f"model-garage v{__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        from model_garage.cli.theme import print_banner
        from model_garage import __version__

        print_banner(__version__)


@app.command()
def open(
    model: str = typer.Argument(help="HuggingFace model ID or local path"),
    device: str = typer.Option("auto", "--device", "-d", help="Device (auto/cuda/cpu)"),
):
    """Load a model and display its architecture card."""
    from model_garage.core.loader import ModelLoader
    from model_garage.cli.theme import console, print_model_card

    with console.status(f"[info]Loading {model}...[/info]"):
        loader = ModelLoader(device=device if device != "auto" else None)
        _, _, info = loader.load(model)

    print_model_card(info)


@app.command()
def extract(
    model: str = typer.Argument(help="HuggingFace model ID or local path"),
    layer: Optional[int] = typer.Option(None, "--layer", "-l", help="Layer index"),
    component: Optional[str] = typer.Option(None, "--component", "-c", help="Component type"),
    all_parts: bool = typer.Option(False, "--all", help="List all components"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Extract components from a model."""
    from model_garage.extract.pytorch import PyTorchExtractor
    from model_garage.cli.theme import console, print_parts_table
    from pathlib import Path

    with console.status(f"[info]Loading {model}...[/info]"):
        extractor = PyTorchExtractor(model)
        extractor.load_model()

    if all_parts or (layer is None and component is None):
        components = extractor.list_available_components()
        console.print(f"\n[success]{len(components)} extractable components[/success]\n")
        print_parts_table(components)
        return

    if layer is not None and component:
        with console.status(f"[info]Extracting {component} from layer {layer}...[/info]"):
            extracted = extractor.extract_component(component, layer_idx=layer)
        console.print(f"[success]Extracted {extracted.spec.name}[/success]")
        console.print(f"  Parameters: {sum(p.numel() for p in extracted.parameters()):,}")
        if output:
            import torch

            out_path = Path(output)
            out_path.mkdir(parents=True, exist_ok=True)
            torch.save(extracted.state_dict, out_path / f"{extracted.spec.name}.pt")
            console.print(f"  Saved to: {out_path / f'{extracted.spec.name}.pt'}")
    elif layer is not None:
        with console.status(f"[info]Extracting layer {layer}...[/info]"):
            components = extractor.extract_layer(layer)
        console.print(f"\n[success]{len(components)} components from layer {layer}[/success]")
        for name, comp in components.items():
            console.print(f"  {name}: {sum(p.numel() for p in comp.parameters()):,} params")


@app.command()
def analyze(
    model: str = typer.Argument(help="HuggingFace model ID or local path"),
    prompt: str = typer.Option("The quick brown fox", "--prompt", "-p", help="Input text"),
):
    """Analyze model activations across layers."""
    import torch
    from model_garage.core.loader import ModelLoader
    from model_garage.core.hooks import HookManager
    from model_garage.core.tensor import TensorUtils
    from model_garage.cli.theme import console
    from rich.table import Table

    with console.status(f"[info]Loading {model}...[/info]"):
        loader = ModelLoader()
        mdl, tokenizer, info = loader.load(model)

    inputs = tokenizer(prompt, return_tensors="pt").to(next(mdl.parameters()).device)
    layer_names = loader.get_layer_names(mdl)
    layers = layer_names.get("layers", [])

    hook_mgr = HookManager(mdl)
    for ln in layers:
        hook_mgr.register_capture_hook(ln, hook_name=ln)

    with torch.no_grad():
        mdl(**inputs)

    table = Table(title="Activation Analysis", border_style="cyan")
    table.add_column("Layer", style="layer")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Sparsity", justify="right")

    for ln in layers:
        data = hook_mgr.get_captured(ln)
        if data and "output" in data:
            stats = TensorUtils.stats(data["output"])
            label = ln.split(".")[-1] if "." in ln else ln
            table.add_row(label, f"{stats['mean']:.4f}", f"{stats['std']:.4f}", f"{stats['sparsity']:.2%}")

    hook_mgr.remove_all()
    console.print(table)


@app.command()
def compare(
    model_a: str = typer.Argument(help="First model"),
    model_b: str = typer.Argument(help="Second model"),
):
    """Compare two model architectures for compatible parts."""
    from model_garage.core.loader import ModelLoader
    from model_garage.registry.models import ModelRegistry
    from model_garage.cli.theme import console
    from rich.table import Table

    loader = ModelLoader()
    registry = ModelRegistry()

    with console.status(f"[info]Loading {model_a}...[/info]"):
        mdl_a, _, _ = loader.load(model_a)
        registry.register(model_a, mdl_a)

    with console.status(f"[info]Loading {model_b}...[/info]"):
        mdl_b, _, _ = loader.load(model_b)
        registry.register(model_b, mdl_b)

    comparison = registry.compare(model_a, model_b)

    table = Table(title="Architecture Comparison", border_style="cyan")
    table.add_column("Property", style="dim")
    table.add_column(model_a.split("/")[-1], style="cyan", justify="right")
    table.add_column(model_b.split("/")[-1], style="green", justify="right")

    for key in ["families", "hidden_dims", "num_layers", "num_heads", "vocab_sizes"]:
        vals = comparison[key]
        table.add_row(key.replace("_", " ").title(), str(vals[0]), str(vals[1]))

    console.print(table)

    if comparison.get("compatible_parts", {}).get("same_dim"):
        console.print("\n[success]Same hidden dimension - parts are swappable![/success]")
    else:
        console.print("\n[warning]Different dimensions - adapters needed[/warning]")


if __name__ == "__main__":
    app()
