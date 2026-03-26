# Contributing

The most impactful contribution you can make is **adding support for a new model family**.

## Adding a New Model Family

Model Garage uses `ModelDecomposer` classes to understand different architectures.

### 1. Create a Decomposer

In `src/model_garage/registry/models.py`, add a new decomposer class:

```python
class MyModelDecomposer(ModelDecomposer):
    """Decomposer for MyModel family."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return "mymodel" in model_id.lower()

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        config = model.config
        spec = ModelSpec(
            model_id=model_id,
            family=ModelFamily.MYMODEL,  # Add to enum first
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_position_embeddings,
        )
        # Add parts (embedding, attention, ffn, norms, output head)
        # See existing decomposers for patterns
        return spec

    def get_module(self, model: nn.Module, part: PartSpec) -> nn.Module:
        module = model
        for attr in part.module_path.split("."):
            module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
        return module
```

### 2. Add Architecture Pattern

In `src/model_garage/extract/pytorch.py`, add an entry to `ARCH_PATTERNS`:

```python
'mymodel': {
    'layers_path': 'model.layers',
    'attention': 'self_attn',
    'ffn': 'mlp',
    'input_norm': 'input_layernorm',
    'post_attn_norm': 'post_attention_layernorm',
    'embed': 'model.embed_tokens',
    'lm_head': 'lm_head',
},
```

### 3. Register the Decomposer

Add your decomposer to `ModelRegistry.__init__()`:

```python
self.decomposers = [
    ...,
    MyModelDecomposer(),
]
```

### 4. Test It

```bash
python -c "
from model_garage.extract.pytorch import PyTorchExtractor
e = PyTorchExtractor('my-org/my-model')
e.load_model()
print(e.summary())
print(len(e.list_available_components()), 'components')
"
```

## Development Setup

```bash
git clone https://github.com/Lumi-node/model-garage
cd model-garage
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

We use `ruff` for linting:

```bash
ruff check src/
ruff format src/
```

## Building Docs Locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000).

## Pull Request Guidelines

- One PR per model family or feature
- Include a test that loads a small model and extracts at least one component
- Update the README's supported models table
- Run `pytest tests/` and `ruff check src/` before submitting
