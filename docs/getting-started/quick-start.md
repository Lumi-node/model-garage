# Quick Start

Your first model surgery in 5 minutes.

## 1. Open a Model

=== "Python"

    ```python
    from model_garage import ModelLoader

    loader = ModelLoader()
    model, tokenizer, info = loader.load("gpt2")
    print(f"Loaded {info['model_id']}: {info['total_params']:,} parameters")
    ```

=== "CLI"

    ```bash
    garage open gpt2
    ```

## 2. Decompose Into Parts

```python
from model_garage import ModelRegistry

registry = ModelRegistry()
spec = registry.register("gpt2", model)

print(f"{len(spec.parts)} extractable parts:")
for name, part in list(spec.parts.items())[:5]:
    print(f"  {name}: {part.part_type.value} [{part.input_dim}d]")
```

The registry analyzes the model architecture and catalogs every extractable component — attention heads, FFN layers, embeddings, normalization layers, and the output head.

## 3. Extract a Component

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

!!! tip "Real Modules"
    Extracted components are real `nn.Module` objects with full parameters.
    You can run them, analyze them, save them, or transplant them into another model.

## 4. Inject Modifications

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

Injections are applied as context managers — they modify the forward pass temporarily and clean up automatically when the `with` block exits.

## 5. Capture Hidden States

```python
from model_garage.snapshot.capture import SnapshotCapture

capture = SnapshotCapture(model)
snapshots = capture.run(
    input_ids,
    layers=["transformer.h.0", "transformer.h.6", "transformer.h.11"]
)

for name, snap in snapshots.items():
    print(f"{name}: mean={snap.mean_activation:.4f}, sparsity={snap.sparsity:.2%}")
```

## 6. Compare Models

```python
model_b, _, _ = loader.load("distilgpt2")
registry.register("distilgpt2", model_b)

comparison = registry.compare("gpt2", "distilgpt2")
print(f"Same hidden dim: {comparison['hidden_dims'][0] == comparison['hidden_dims'][1]}")
print(f"Compatible: {comparison['compatible_parts']}")
```

## Next Steps

- Learn about the [core concepts](concepts.md) behind the library
- Deep dive into [extraction](../guides/extraction.md), [injection](../guides/injection.md), or [analysis](../guides/analysis.md)
- Read the [research papers](../research/index.md) for validated methodologies
- See the [examples/](https://github.com/Lumi-node/model-garage/tree/main/examples) directory for complete scripts
