# Research Papers

Three peer-reviewed papers demonstrating Model Garage capabilities on real problems. Full LaTeX source and compiled PDFs included for transparency and reproducibility.

## Papers

### Blades: Compositional Capability Enhancement Through Hidden State Injection

**[PDF](blades/main.pdf)** | **[LaTeX Source](blades/main.tex)**

Hidden state injection between specialized models achieves **+14.2% accuracy** on medical reasoning tasks. Establishes 7 validated principles for capability transfer including the N-4 layer rule and same-domain synergy (+27.8%).

*Model Garage modules used: `extract`, `inject`, `snapshot`, `core.hooks`*

---

### Learned Routers Don't Learn: Expert Miscalibration in MoE Models

**[PDF](moe-miscalibration/main.pdf)** | **[LaTeX Source](moe-miscalibration/main.tex)**

Per-layer expert isolation reveals that learned MoE routers show **Spearman rho ~ 0** between routing probability and expert quality. 207/896 expert-layer-domain combinations show statistically significant specialization (BH FDR, alpha=0.05), yet the router ignores it.

*Model Garage modules used: `analyze`, `core.hooks`, `registry`*

---

### Sparse Pathways: Domain-Aware Neuron Routing for Efficient Inference

**[PDF](sparse-pathways/main.pdf)** | **[LaTeX Source](sparse-pathways/main.tex)**

FFN neurons exhibit strong domain specialization (~50% in late layers), with **r=0.999 correlation** between model scale and specialization degree. Demonstrates **2-4x potential compute reduction** via negative neuron selection.

*Model Garage modules used: `analyze`, `snapshot`, `core.hooks`*

---

## Citation

If you use these findings or Model Garage in your research, please cite:

```bibtex
@software{model_garage,
  title = {Model Garage: Component-Level Neural Network Surgery},
  author = {Model Garage Contributors},
  year = {2026},
  url = {https://github.com/Lumi-node/model-garage},
  license = {Apache-2.0}
}
```
