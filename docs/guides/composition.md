# Model Composition

Register multiple models in the registry and find compatible parts for building hybrid architectures.

## Register Models

```python
from model_garage.core.loader import ModelLoader
from model_garage.registry.models import ModelRegistry

loader = ModelLoader()
registry = ModelRegistry()

# Register two models
model_a, _, _ = loader.load("gpt2")
spec_a = registry.register("gpt2", model_a)

model_b, _, _ = loader.load("distilgpt2")
spec_b = registry.register("distilgpt2", model_b)

print(f"gpt2: {spec_a.hidden_dim}d, {spec_a.num_layers}L, {len(spec_a.parts)} parts")
print(f"distilgpt2: {spec_b.hidden_dim}d, {spec_b.num_layers}L, {len(spec_b.parts)} parts")
```

## Compare Architectures

```python
comparison = registry.compare("gpt2", "distilgpt2")

print(f"Hidden dims: {comparison['hidden_dims']}")
print(f"Layers: {comparison['num_layers']}")
print(f"Compatible parts: {comparison['compatible_parts']}")
```

Two components are **compatible** when they share the same:

- Input dimension
- Output dimension
- Component type (attention, FFN, etc.)

## Finding Compatible Parts

```python
for part in comparison['compatible_parts']:
    print(f"  {part['type']}: {part['name_a']} <-> {part['name_b']}")
```

!!! warning "Same-Dimension Requirement"
    Components from models with different hidden dimensions cannot be directly swapped.
    See the [Blades paper](../research/blades.md) for projection-based alternatives (Rule 7: FFN Projection Works).

## ModelSpec Inspection

Explore what's inside a registered model:

```python
spec = registry.get("gpt2")

for name, part in spec.parts.items():
    print(f"{name}: {part.part_type.value}")
    print(f"  dims: [{part.input_dim} -> {part.output_dim}]")
    print(f"  layer: {part.layer_idx}")
```

## CLI

```bash
# Compare two models
garage compare gpt2 distilgpt2

# List registered models
garage registry list

# Register a new model
garage registry add microsoft/phi-2
```

## Next Steps

- [Extract](extraction.md) compatible components for transplant
- [Inject](injection.md) extracted parts into a target model
- Read about [capability transfer principles](../research/blades.md)
