# Activation Analysis

<figure markdown="span">
  ![Neural Net Diagnostic Scanner](../assets/images/diagnostic_scanner.jpg){ width="700" loading=lazy }
</figure>

Capture and inspect hidden states at any layer. Understand what each layer contributes through activation statistics, entropy measurement, and sparsity analysis.

## Hook-Based Capture

The low-level API uses hooks to intercept the forward pass:

```python
from model_garage.core.hooks import HookManager
from model_garage.core.tensor import TensorUtils

with HookManager(model) as hooks:
    hooks.register_capture_hook("model.layers.15", hook_name="layer_15")
    model(input_ids)

    data = hooks.get_captured("layer_15")
    stats = TensorUtils.stats(data["output"])
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std:  {stats['std']:.4f}")
    print(f"Sparsity: {stats['sparsity']:.2%}")
```

## Snapshot Capture

The high-level API captures multiple layers at once with computed statistics:

```python
from model_garage.snapshot.capture import SnapshotCapture

capture = SnapshotCapture(model)
snapshots = capture.run(
    input_ids,
    layers=["transformer.h.0", "transformer.h.6", "transformer.h.11"]
)

for name, snap in snapshots.items():
    print(f"{name}:")
    print(f"  Mean activation: {snap.mean_activation:.4f}")
    print(f"  Sparsity: {snap.sparsity:.2%}")
```

## Tensor Utilities

`TensorUtils` provides common analysis operations:

```python
from model_garage.core.tensor import TensorUtils

# Full statistics
stats = TensorUtils.stats(tensor)
# Returns: mean, std, min, max, sparsity, norm, shape

# Cosine similarity between two hidden states
similarity = TensorUtils.cosine_sim(hidden_a, hidden_b)
```

## Layer-by-Layer Analysis

Capture every layer to understand the transformation pipeline:

```python
all_layers = [f"transformer.h.{i}" for i in range(model.config.n_layer)]

capture = SnapshotCapture(model)
snapshots = capture.run(input_ids, layers=all_layers)

# Plot sparsity progression
for name, snap in snapshots.items():
    layer_num = int(name.split('.')[-1])
    bar = '#' * int(snap.sparsity * 50)
    print(f"  Layer {layer_num:2d}: {bar} {snap.sparsity:.1%}")
```

## CLI

```bash
# Analyze activations for a prompt
garage analyze gpt2 --prompt "The meaning of life is"

# Analyze specific layers
garage analyze gpt2 --prompt "Hello world" --layers 0,6,11
```

## Use Cases

- **Interpretability** — Understand what each layer learns
- **Debugging** — Find layers producing anomalous activations
- **Pruning** — Identify high-sparsity layers that can be compressed
- **Research** — Validate hypotheses about model behavior (see [Sparse Pathways](../research/sparse-pathways.md))

## Next Steps

- [Extract](extraction.md) components from interesting layers
- [Inject](injection.md) modifications and measure their effect
- Read the [Sparse Pathways paper](../research/sparse-pathways.md) for domain-specific neuron analysis
