# CLI Reference

Model Garage provides a retro-themed CLI built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

## Installation

The CLI is installed automatically with the package:

```bash
pip install model-garage
garage --help
```

## Commands

### `garage open`

Load and display a model's architecture card.

```bash
garage open gpt2
garage open microsoft/phi-2
garage open meta-llama/Llama-2-7b-hf
```

Shows:

- Model family and architecture type
- Parameter count and layer structure
- Hidden dimensions, attention heads, vocabulary size
- Extractable component summary

### `garage extract`

Extract components from a model.

```bash
# Extract attention from a specific layer
garage extract gpt2 --layer 6 --component self_attention

# Extract FFN from layer 3
garage extract gpt2 --layer 3 --component ffn

# Extract all components from a layer
garage extract gpt2 --layer 6
```

**Options:**

| Flag | Description |
|------|-------------|
| `--layer`, `-l` | Layer index to extract from |
| `--component`, `-c` | Component type: `self_attention`, `ffn`, `embedding`, `layer_norm` |
| `--output`, `-o` | Output path for saved component |

### `garage analyze`

Analyze activations for a given prompt.

```bash
garage analyze gpt2 --prompt "The meaning of life is"
garage analyze microsoft/phi-2 --prompt "Hello world" --layers 0,6,11
```

Shows activation statistics per layer: mean, std, sparsity, and entropy.

**Options:**

| Flag | Description |
|------|-------------|
| `--prompt`, `-p` | Input text to analyze |
| `--layers` | Comma-separated layer indices (default: all) |

### `garage compare`

Compare two model architectures and find compatible parts.

```bash
garage compare gpt2 distilgpt2
garage compare microsoft/phi-2 microsoft/phi-1_5
```

Shows dimension comparison, layer counts, and lists compatible components.

### `garage registry list`

List all models registered in the local registry.

```bash
garage registry list
```

### `garage registry add`

Decompose and register a model.

```bash
garage registry add gpt2
garage registry add microsoft/phi-2
```

## Output Examples

<figure markdown="span">
  ![garage open](../assets/screenshots/model_card.svg){ width="500" loading=lazy }
  <figcaption>garage open gpt2</figcaption>
</figure>

<figure markdown="span">
  ![garage extract](../assets/screenshots/extract.svg){ width="600" loading=lazy }
  <figcaption>garage extract gpt2 --layer 6 --component self_attention</figcaption>
</figure>

<figure markdown="span">
  ![garage analyze](../assets/screenshots/analyze.svg){ width="580" loading=lazy }
  <figcaption>garage analyze gpt2</figcaption>
</figure>

<figure markdown="span">
  ![garage compare](../assets/screenshots/compare.svg){ width="500" loading=lazy }
  <figcaption>garage compare gpt2 distilgpt2</figcaption>
</figure>
