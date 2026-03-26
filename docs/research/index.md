# Research

Model Garage is validated through three peer-reviewed research papers demonstrating its capabilities on real problems. Each paper uses Model Garage modules directly and reports quantified, reproducible results.

## Papers

<div class="grid cards" markdown>

-   **Blades: Compositional Capability Enhancement Through Hidden State Injection**

    ---

    Hidden state injection between specialized models achieves **+14.2% accuracy** on medical reasoning tasks. Establishes **7 validated principles** for capability transfer including the N-4 layer rule and same-domain synergy (+27.8%).

    *Modules used:* `extract`, `inject`, `snapshot`, `core.hooks`

    [:octicons-arrow-right-24: Full summary](blades.md) | [:material-file-pdf-box: PDF](https://github.com/Lumi-node/model-garage/raw/main/research/papers/blades/Blades_Compositional_Capability_Enhancement_Young2026.pdf)

-   **Learned Routers Don't Learn: Expert Miscalibration in MoE Models**

    ---

    Per-layer expert isolation reveals that learned MoE routers show **Spearman rho = 0.069** between routing probability and expert quality. **482/896** expert-layer-domain combinations show statistically significant specialization, yet the router ignores it.

    *Modules used:* `analyze`, `core.hooks`, `registry`

    [:octicons-arrow-right-24: Full summary](moe-routing.md) | [:material-file-pdf-box: PDF](https://github.com/Lumi-node/model-garage/raw/main/research/papers/moe-miscalibration/Learned_Routers_Dont_Learn_MoE_Miscalibration_Young2026.pdf)

-   **Sparse Pathways: Domain-Aware Neuron Routing for Efficient Inference**

    ---

    FFN neurons exhibit strong domain specialization (~50% in late layers), with **r=0.999 correlation** between model scale and specialization degree. Demonstrates **2-4x potential compute reduction** via negative neuron selection.

    *Modules used:* `analyze`, `snapshot`, `core.hooks`

    [:octicons-arrow-right-24: Full summary](sparse-pathways.md) | [:material-file-pdf-box: PDF](https://github.com/Lumi-node/model-garage/raw/main/research/papers/sparse-pathways/Sparse_Pathways_Domain_Aware_Neuron_Routing_Young2026.pdf)

</div>

## Citation

If you use these findings or Model Garage in your research:

```bibtex
@software{model_garage,
  title = {Model Garage: Component-Level Neural Network Surgery},
  author = {Model Garage Contributors},
  year = {2026},
  url = {https://github.com/Lumi-node/model-garage},
  license = {Apache-2.0}
}
```

## Pretrained Artifacts

The research directory includes pretrained components ready for use:

- **[Extracted Components](https://github.com/Lumi-node/model-garage/tree/main/research/extracted-components)** — Pre-extracted attention heads and FFN modules
- **[Pretrained Blades](https://github.com/Lumi-node/model-garage/tree/main/research/pretrained-blades)** — SAE-trained blades for capability injection
- **[Blade Principles](blades.md#the-7-rules)** — Validated rules for capability transfer
