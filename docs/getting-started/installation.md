# Installation

## From PyPI

```bash
pip install model-garage
```

## With Optional Dependencies

```bash
# Development tools (pytest, ruff, mypy)
pip install model-garage[dev]

# TUI dashboard (Textual)
pip install model-garage[dashboard]

# Everything
pip install model-garage[all]
```

## From Source

```bash
git clone https://github.com/Lumi-node/model-garage
cd model-garage
pip install -e ".[dev]"
```

## Requirements

- **Python** 3.10+
- **PyTorch** 2.0+
- **Transformers** 4.30+

!!! note "GPU Optional"
    Model Garage works on CPU for small models (GPT-2, DistilBERT).
    For larger models (Llama, Phi-4), a CUDA-capable GPU is recommended.

## Verify Installation

```bash
garage --help
```

Or from Python:

```python
import model_garage
print(model_garage.__version__)
```

## Next Steps

- Follow the [Quick Start](quick-start.md) to perform your first model surgery
- Read about [core concepts](concepts.md) behind Model Garage
