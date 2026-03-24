# Pretrained SAE Blades

Trained Sparse Autoencoder (SAE) blades extracted from GPT-2 at key layer positions. These are the actual trained modules from validated experiments — ready to use for capability analysis, injection, and interpretability research.

## Files

| File | Layer | Size | What It Captures |
|------|-------|------|-----------------|
| `sae_layer_0.pt` | 0 (input) | 19MB | Token-level features, embedding patterns |
| `sae_layer_3.pt` | 3 (early) | 19MB | Syntactic features, basic composition |
| `sae_layer_6.pt` | 6 (mid) | 19MB | Semantic features, relational patterns |
| `sae_layer_9.pt` | 9 (late) | 19MB | Abstract reasoning, task-specific features |
| `sae_layer_11.pt` | 11 (final) | 19MB | Output-aligned features, prediction heads |

## Usage

```python
import torch
from model_garage.core.loader import quick_load

# Load the SAE blade
sae = torch.load("research/pretrained-blades/sae_layer_6.pt", map_location="cpu")

# Load the model it was trained on
model, tokenizer, _ = quick_load("gpt2")

# Use with HookManager to analyze activations through the SAE lens
from model_garage.core.hooks import HookManager

with HookManager(model) as hooks:
    hooks.register_capture_hook("transformer.h.6", hook_name="layer6")
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    hidden = hooks.get_captured("layer6")["output"]

    # Encode through SAE to find active features
    # (exact API depends on SAE architecture)
```

## Training Details

- **Source model**: GPT-2 (124M parameters, 768 hidden dim)
- **Training data**: Diverse text corpus
- **Architecture**: Sparse autoencoder with learned dictionary
- **Sparsity**: L1 regularization for interpretable features

## Full Models

For full decomposed models with all components (FFN, embeddings, full layers), see our [HuggingFace Hub](https://huggingface.co/Lumi-node).
