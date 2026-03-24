# Extracted Components

Real `nn.Module` components extracted from transformer models using Model Garage. These are actual PyTorch weights — not approximations or metadata.

## Files

### gpt2-attention-L6/

Attention mechanism extracted from GPT-2, layer 6.

| Property | Value |
|----------|-------|
| Source model | gpt2 (124M params) |
| Component | Self-attention |
| Layer | 6 of 12 |
| Hidden dim | 768 |
| Num heads | 12 |
| Head dim | 64 |
| Parameters | 2,362,368 |
| Size | 9.1MB |

**Files:**
- `weights.pt` — The actual attention module weights
- `config.json` — Component metadata (dimensions, source, layer)

## Usage

```python
import torch

# Load the extracted attention
state_dict = torch.load(
    "research/extracted-components/gpt2-attention-L6/weights.pt",
    map_location="cpu"
)

# Or use the extract API to get a live module
from model_garage.extract.pytorch import PyTorchExtractor

extractor = PyTorchExtractor("gpt2")
extractor.load_model()
attn = extractor.extract_component("self_attention", layer_idx=6)

# Test it in isolation
from model_garage.extract.pytorch import ComponentTester
tester = ComponentTester(device="cpu")
result = tester.test_attention(attn)
print(f"Test: {'PASS' if result['success'] else 'FAIL'}")
```

## Want More Components?

Use the CLI to extract from any supported model:

```bash
garage extract gpt2 --layer 6 --component self_attention --output ./my_components/
garage extract microsoft/phi-2 --all
```

For full decomposed models, see our [HuggingFace Hub](https://huggingface.co/Lumi-node).
