# Layer Injection

Insert custom processing between any two layers without modifying the model. Injections are applied as context managers and clean up automatically.

## Scaling Injection

Multiply activations by a constant factor:

```python
import torch
from model_garage.core.loader import quick_load
from model_garage.inject.layer import LayerInjector

model, tokenizer, info = quick_load("gpt2")
input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids

# Scale activations at layer 6 by 90%
with LayerInjector(model) as injector:
    injector.inject_scaling("transformer.h.6", scale=0.9)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

!!! tip "Auto-Cleanup"
    When the `with` block exits, all injections are removed and the model is restored to its original state.

## Custom Module Injection

Inject an arbitrary `nn.Module` between layers:

```python
import torch.nn as nn

class ActivationNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

with LayerInjector(model) as injector:
    injector.inject_custom_layer("transformer.h.6", ActivationNoise(std=0.02))
    output = model(input_ids)
```

## Blade Injection

Blades are pre-trained hidden state modifications that transfer capabilities between models. See the [Blades research paper](../research/blades.md) for the validated principles.

```python
from model_garage.inject.debate import SelfDebate

# Use gated injection for best results (+8.9% over identity)
with SelfDebate(model, layer_idx=6, reconciliation_method="gated"):
    output = model.generate(input_ids, max_new_tokens=50)
```

### The 7 Rules of Blade Injection

1. **N-4 Layer Rule** — Inject at layer N-4 (87.5% depth) for best results
2. **Same-Dimension Requirement** — Source and target must share hidden dimensions
3. **Capability Gap Principle** — Benefit is proportional to the gap between source and target
4. **Gated > Identity** — Learned gating outperforms direct injection by +8.9%
5. **Same-Domain Synergy** — Multiple blades from the same domain synergize (+27.8%)
6. **MoE Router Control** — Router bias enables domain-selective expert activation
7. **FFN Projection Works** — High-dimensional FFN outputs can be projected down

## Multiple Injections

Stack multiple injections in a single context:

```python
with LayerInjector(model) as injector:
    injector.inject_scaling("transformer.h.4", scale=1.1)
    injector.inject_scaling("transformer.h.8", scale=0.85)
    injector.inject_custom_layer("transformer.h.6", my_adapter)
    output = model(input_ids)
```

## CLI

```bash
# Not yet exposed via CLI — use Python API
```

## Next Steps

- Learn about [Self-Debate](debate.md) chambers for multi-perspective reasoning
- Read the [Blades paper](../research/blades.md) for capability transfer methodology
- [Analyze](analysis.md) injection effects with activation capture
