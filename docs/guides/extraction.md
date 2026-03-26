# Component Extraction

<figure markdown="span">
  ![Extraction Surgery](../assets/images/extraction_surgery.jpg){ width="700" loading=lazy }
</figure>

Pull real `nn.Module` components from any supported transformer model. Extracted components retain their full parameters and can be run, analyzed, saved, or transplanted into another model.

## Basic Extraction

```python
from model_garage.extract.pytorch import PyTorchExtractor

extractor = PyTorchExtractor("gpt2")
extractor.load_model()

# See what's available
print(f"Architecture: {extractor.arch_type}")
print(f"Layers: {extractor.get_num_layers()}")
print(f"Hidden size: {extractor.get_hidden_size()}")
```

## Extract Specific Components

### Attention Heads

```python
attn = extractor.extract_component("self_attention", layer_idx=6)
print(f"Parameters: {sum(p.numel() for p in attn.parameters()):,}")

# It's a real module — run it
output = attn.module(hidden_states)
```

### Feed-Forward Networks

```python
ffn = extractor.extract_component("ffn", layer_idx=6)
```

### Embeddings

```python
embed = extractor.extract_component("embedding")
```

## Testing Extracted Components

Always verify that extracted components produce valid outputs:

```python
from model_garage.extract.pytorch import ComponentTester

tester = ComponentTester()
result = tester.test_attention(attn)
print(f"Test: {'PASS' if result['success'] else 'FAIL'}")
print(f"Output shape: {result['output_shape']}")
```

## List Available Components

```python
components = extractor.list_available_components()
for comp in components:
    print(f"  {comp['name']}: {comp['type']} (layer {comp.get('layer', 'N/A')})")
```

## Architecture Patterns

Model Garage uses architecture patterns to navigate different model families. Each pattern maps component names to module paths:

```python
# Example: GPT-2 pattern
{
    'layers_path': 'transformer.h',
    'attention': 'attn',
    'ffn': 'mlp',
    'input_norm': 'ln_1',
    'post_attn_norm': 'ln_2',
    'embed': 'transformer.wte',
    'lm_head': 'lm_head',
}
```

## CLI

```bash
# Extract attention from layer 6
garage extract gpt2 --layer 6 --component self_attention

# Extract all components from a layer
garage extract gpt2 --layer 6

# Extract from any supported model
garage extract microsoft/phi-2 --layer 12 --component self_attention
```

## Saving & Loading Components

Extracted components can be saved and reloaded:

```python
import torch

# Save
torch.save({
    'module': attn.module.state_dict(),
    'spec': attn.spec,
}, 'gpt2_attn_L6.pt')

# Load later
checkpoint = torch.load('gpt2_attn_L6.pt')
```

## Next Steps

- [Inject](injection.md) extracted components into other models
- [Analyze](analysis.md) component behavior with activation capture
- [Compose](composition.md) hybrid models from extracted parts
