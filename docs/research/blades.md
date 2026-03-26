# Blades: Compositional Capability Enhancement

**Paper:** [Blades: Compositional Capability Enhancement Through Hidden State Injection](https://github.com/Lumi-node/model-garage/raw/main/research/papers/blades/Blades_Compositional_Capability_Enhancement_Young2026.pdf)

## Summary

Hidden state injection between specialized models achieves **+14.2% accuracy** on medical reasoning tasks. The paper establishes **7 validated principles** for capability transfer, tested through 17 experiments with quantified results.

## Key Finding

A "blade" is a hidden state modification extracted from a specialist model and injected into a generalist model at a specific layer. This transfers capability without fine-tuning.

```
Source model (specialist) → Extract hidden state → Inject into target → Improved capability
```

## The 7 Rules of Capability Transfer { #the-7-rules }

### 1. N-4 Layer Rule

Optimal injection point is at layer N-4 (87.5% depth).

| Injection Point | Success Rate |
|-----------------|-------------|
| Layer 4 (early) | 0% |
| Layer N-4 (late) | **70%** (peak) |
| Layer N+ (too late) | Degraded |

### 2. Same-Dimension Requirement

Source and target models must share the same hidden dimension for direct transfer. Dimension mismatch causes information loss (e.g., 3072 -> 640 = -8.2% accuracy).

### 3. Capability Gap Principle

Blade benefit is proportional to the capability gap between source and target:

$$\text{improvement} \propto (\text{source\_capability} - \text{target\_capability})$$

Injecting into a model that's already stronger causes degradation (-6.2%).

### 4. Gated > Identity

Learned gating mechanisms outperform direct injection by **+8.9%**. Always use gated injection for production transfers.

### 5. Same-Domain Synergy

Multiple blades from the same domain synergize (**+27.8%**). Cross-domain blades interfere (**-27.8%**).

### 6. MoE Router Control

Router bias (strength 5-10) enables domain-selective expert activation. Achieves **1.67x selectivity** improvement for targeted expert routing.

### 7. FFN Projection Works

High-dimensional FFN outputs (14336d) can be projected to lower dimensions (960d) using magnitude-based truncation while maintaining functionality.

## Best Results

| Source | Target | Capability | Result | Method |
|--------|--------|------------|--------|--------|
| Phi-4-reasoning | MediPhi | Reasoning -> Medical | **+14.2%** | Gated @ L28 |
| Layer sweep | MediPhi | Various | **70% @ L28** | N-4 rule |
| rho-math-7b | Phi-mini-MoE | Router control | **1.67x selectivity** | Directional bias |

## Using Blades in Model Garage

```python
from model_garage.inject.layer import LayerInjector

# Load a pre-trained blade
import torch
blade = torch.load("research/pretrained-blades/sae_layer_6.pt")

# Apply the N-4 rule: for a 32-layer model, inject at layer 28
with LayerInjector(model) as injector:
    injector.inject_custom_layer("model.layers.28", blade)
    output = model.generate(input_ids, max_new_tokens=50)
```

## Model Garage Modules Used

- `extract` — Component extraction from source models
- `inject` — Hidden state injection into target models
- `snapshot` — Capturing hidden states for blade training
- `core.hooks` — Forward pass interception
