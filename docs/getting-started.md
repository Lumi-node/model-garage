# Getting Started

## Installation

```bash
pip install model-garage
```

For development:
```bash
git clone https://github.com/model-garage/model-garage
cd model-garage
pip install -e ".[dev]"
```

## Your First Surgery

### 1. Open a model

```python
from model_garage import ModelLoader

loader = ModelLoader()
model, tokenizer, info = loader.load("gpt2")
print(f"Loaded {info['model_id']}: {info['total_params']:,} parameters")
```

Or from the CLI:
```bash
garage open gpt2
```

### 2. Decompose it into parts

```python
from model_garage import ModelRegistry

registry = ModelRegistry()
spec = registry.register("gpt2", model)

print(f"{len(spec.parts)} extractable parts:")
for name, part in list(spec.parts.items())[:5]:
    print(f"  {name}: {part.part_type.value} [{part.input_dim}d]")
```

### 3. Extract a component

```python
from model_garage.extract.pytorch import PyTorchExtractor

extractor = PyTorchExtractor("gpt2")
extractor.load_model()

# Pull out attention from layer 6
attn = extractor.extract_component("self_attention", layer_idx=6)
print(f"Extracted: {sum(p.numel() for p in attn.parameters()):,} parameters")

# It's a real nn.Module — test it in isolation
from model_garage.extract.pytorch import ComponentTester
tester = ComponentTester()
result = tester.test_attention(attn)
print(f"Test: {'PASS' if result['success'] else 'FAIL'}")
```

### 4. Inject modifications

```python
import torch
from model_garage.inject.layer import LayerInjector

input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids

# Inject scaling at layer 6
with LayerInjector(model) as injector:
    injector.inject_scaling("transformer.h.6", scale=0.9)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

### 5. Capture hidden states

```python
from model_garage.snapshot.capture import SnapshotCapture

capture = SnapshotCapture(model)
snapshots = capture.run(input_ids, layers=["transformer.h.0", "transformer.h.6", "transformer.h.11"])

for name, snap in snapshots.items():
    print(f"{name}: mean={snap.mean_activation:.4f}, sparsity={snap.sparsity:.2%}")
```

### 6. Compare models

```python
model_b, _, _ = loader.load("distilgpt2")
registry.register("distilgpt2", model_b)

comparison = registry.compare("gpt2", "distilgpt2")
print(f"Same hidden dim: {comparison['hidden_dims'][0] == comparison['hidden_dims'][1]}")
print(f"Compatible: {comparison['compatible_parts']}")
```

## Next Steps

- See [examples/](../examples/) for more complete scripts
- Read [BLADE_PRINCIPLES.md](../research/BLADE_PRINCIPLES.md) for capability transfer rules
- Check [philosophy.md](philosophy.md) for the "Gear Head" design philosophy
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to add support for new model families
