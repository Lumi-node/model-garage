# Sparse Pathways: Domain-Aware Neuron Routing

**Paper:** [Sparse Pathways: Domain-Aware Neuron Routing for Efficient Inference](https://github.com/Lumi-node/model-garage/raw/main/research/papers/sparse-pathways/Sparse_Pathways_Domain_Aware_Neuron_Routing_Young2026.pdf)

## Summary

FFN neurons exhibit strong domain specialization (~50% in late layers), with **r=0.999 correlation** between model scale and specialization degree. The paper demonstrates **2-4x potential compute reduction** via negative neuron selection — skipping neurons that hurt performance on a given domain.

## Key Finding

Individual FFN neurons are not general-purpose. Many neurons activate strongly for specific domains (medical, legal, code) and weakly or negatively for others. By identifying and skipping domain-negative neurons, you can reduce compute while maintaining or improving accuracy.

```
All neurons active:    100% compute, baseline accuracy
Domain-aware routing:  25-50% compute, maintained accuracy
```

## Key Results

- **~50% of late-layer neurons** show domain specialization
- **r = 0.999** correlation between model scale and neuron specialization degree
- Larger models have more specialized neurons (more "experts" emerge naturally)
- **2-4x potential compute reduction** by skipping domain-negative neurons
- Negative selection (removing harmful neurons) outperforms positive selection (keeping helpful ones)

## Scale-Specialization Correlation

| Model Size | Specialization Degree |
|------------|----------------------|
| Small (125M) | Low |
| Medium (1.3B) | Moderate |
| Large (7B+) | High (~50% in late layers) |

The correlation is near-perfect (r = 0.999), suggesting specialization is an emergent property of scale.

## Negative Neuron Selection

The key insight: rather than finding neurons that *help* a domain (positive selection), it's more effective to find neurons that *hurt* a domain (negative selection) and skip them.

```python
# Conceptual approach using Model Garage
from model_garage.snapshot.capture import SnapshotCapture
from model_garage.core.hooks import HookManager

# Capture neuron activations per domain
capture = SnapshotCapture(model)
medical_snapshots = capture.run(medical_inputs, layers=all_ffn_layers)
general_snapshots = capture.run(general_inputs, layers=all_ffn_layers)

# Identify domain-negative neurons
# (neurons whose activation correlates with worse performance on medical tasks)
```

## Implications

1. **Efficient inference** — Skip 50-75% of FFN computation for domain-specific tasks
2. **Dynamic routing** — Route tokens through domain-appropriate neuron subsets at runtime
3. **Model compression** — Prune domain-irrelevant neurons for specialized deployments
4. **Understanding emergence** — Specialization increases predictably with scale

## Model Garage Modules Used

- `analyze` — Per-neuron activation measurement across domains
- `snapshot` — Hidden state capture at FFN granularity
- `core.hooks` — Neuron-level interception during forward passes
