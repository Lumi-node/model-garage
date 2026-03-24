<p align="center">
  <img src="assets/images/hero_garage_floor.jpg" alt="Model Garage" width="900">
</p>

<h1 align="center">Model Garage</h1>

<p align="center">
  <strong>Open the hood on neural networks.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/model-garage/"><img src="https://img.shields.io/pypi/v/model-garage?style=flat-square&color=cyan" alt="PyPI"></a>
  <a href="https://github.com/model-garage/model-garage/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/model-garage?style=flat-square" alt="Python"></a>
  <img src="https://img.shields.io/badge/models-70%2B%20across%2018%20vendors-green?style=flat-square" alt="Models">
  <img src="https://img.shields.io/badge/tests-77%20passing-brightgreen?style=flat-square" alt="Tests">
</p>

---

Component-level model surgery, analysis, and composition. Extract attention heads, swap FFN layers between models, inject capability blades, and build hybrid architectures — all from a beautiful CLI or Python API.

People used to be "Gear Heads" — they could fine-tune actual car engines manually. Model Garage brings that same hands-on philosophy to neural networks. Open the hood, understand the parts, swap what you need.

---

## Quick Start

```bash
pip install model-garage
```

```bash
# Inspect a model's architecture
garage open gpt2

# Extract attention from layer 6
garage extract gpt2 --layer 6 --component self_attention

# Compare two models for compatible parts
garage compare gpt2 distilgpt2

# Analyze activations across layers
garage analyze gpt2 --prompt "The meaning of life is"
```

<details>
<summary><strong>See CLI output examples</strong></summary>

<p align="center"><img src="assets/screenshots/model_card.svg" alt="garage open" width="500"></p>
<p align="center"><img src="assets/screenshots/extract.svg" alt="garage extract" width="600"></p>
<p align="center"><img src="assets/screenshots/analyze.svg" alt="garage analyze" width="580"></p>
<p align="center"><img src="assets/screenshots/compare.svg" alt="garage compare" width="500"></p>

</details>

Or use the Python API directly:

```python
from model_garage import ModelLoader, ModelRegistry

# Load and decompose
loader = ModelLoader()
model, tokenizer, info = loader.load("gpt2")

registry = ModelRegistry()
spec = registry.register("gpt2", model)

# See all parts
for name, part in spec.parts.items():
    print(f"{name}: {part.part_type.value} [{part.input_dim}→{part.output_dim}]")
```

## What Can You Do?

### Extract

<p align="center">
  <img src="assets/images/extraction_surgery.jpg" alt="Extracting attention from a model" width="700">
</p>

Pull real `nn.Module` components from any supported transformer:

```python
from model_garage.extract.pytorch import PyTorchExtractor

extractor = PyTorchExtractor("microsoft/phi-2")
extractor.load_model()

# Extract attention from layer 12
attn = extractor.extract_component("self_attention", layer_idx=12)

# It's a real module — run it, analyze it, transplant it
output = attn.module(hidden_states)
```

### Inject

Insert custom processing between any two layers without modifying the model:

```python
from model_garage.inject.layer import LayerInjector

with LayerInjector(model) as injector:
    # Scale activations (reduce by 10%)
    injector.inject_scaling("model.layers.12", scale=0.9)

    # Or inject a custom module
    injector.inject_custom_layer("model.layers.12", my_adapter)

    # Model now uses your injection during forward pass
    output = model(input_ids)
# Injections automatically cleaned up
```

### Analyze

<p align="center">
  <img src="assets/images/diagnostic_scanner.jpg" alt="Neural Net Diagnostic Scanner" width="700">
</p>

Capture and inspect hidden states at any layer:

```python
from model_garage.core.hooks import HookManager
from model_garage.core.tensor import TensorUtils

with HookManager(model) as hooks:
    hooks.register_capture_hook("model.layers.15", hook_name="layer_15")
    model(input_ids)

    data = hooks.get_captured("layer_15")
    stats = TensorUtils.stats(data["output"])
    print(f"Mean: {stats['mean']:.4f}, Sparsity: {stats['sparsity']:.2%}")
```

### Compose

Register multiple models and find compatible parts for hybrid architectures:

```python
from model_garage.registry.models import ModelRegistry

registry = ModelRegistry()
registry.register("gpt2", model_a)
registry.register("distilgpt2", model_b)

# Find what's swappable
comparison = registry.compare("gpt2", "distilgpt2")
print(comparison["compatible_parts"])
```

### Debate

Add multi-perspective reasoning to any model via self-debate chambers:

```python
from model_garage.inject.debate import SelfDebate

# Two perspectives reconciled through learned gating
with SelfDebate(model, layer_idx=6, reconciliation_method="gated"):
    output = model.generate(input_ids, max_new_tokens=50)
```

## Supported Architectures

### Decomposition Families (Full Surgery Support)

| Family | Models | Capabilities |
|--------|--------|-------------|
| **GPT-2** | gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2 | Extract, inject, analyze, compose |
| **Llama** | Llama-2-7b, Llama-3-8b, TinyLlama, CodeLlama | Extract, inject, analyze, compose |
| **Phi** | Phi-2, Phi-3.5, Phi-4, Phi-4-reasoning, MediPhi, Fara-7B | Extract, inject, analyze, compose |
| **Phi-MoE** | Phi-3.5-MoE, Phi-mini-MoE, Phi-tiny-MoE | Extract, inject, MoE routing, blade injection |
| **Mistral** | Mistral-7B, Mixtral-8x7B | Extract, inject, analyze, compose |
| **Gemma** | Gemma-2b/7b, Gemma-3, FunctionGemma, MedGemma, VaultGemma | Extract, inject, analyze, compose |
| **Qwen** | Qwen-1.5, Qwen-2, Kimi-K2.5 | Extract, inject, analyze, compose |
| **BERT/Encoders** | bert-base/large, distilbert, MiniLM, mpnet | Extraction, analysis |
| **Protein** | ESM2 (8M to 3B) | Extraction, analysis |
| **BitNet** | bitnet-b1.58-2B-4T | Extraction, analysis |

### Models in Repository (70+ across 18 vendors)

Decomposed and ready for surgery from: Alibaba, ByteDance, Facebook/Meta, Google, LightOn, Microsoft, Moonshot, NVIDIA, Sesame, StepFun, Tencent, Wan-AI, and more. Includes language, vision, medical, protein, 3D, audio, video, OCR, and robotics models.

### Zero-Training Expert Router (Rust)

<p align="center">
  <img src="assets/images/router_switchboard.jpg" alt="Zero-Training Expert Router" width="700">
</p>

Model Garage includes a **zero-training semantic router** for MoE and multi-expert systems:

- **No training required** — routes via embedding similarity against expert manifests
- **Sub-2ms routing** at 500+ queries/sec (Python) with Rust core for production
- **DAG execution** — wave-based parallel dispatch for multi-expert pipelines
- **Complexity detection** — automatic simple vs. complex path selection
- **OS-style extensibility** — add an expert by dropping a `manifest.json`

```
Query → Embedding → Cosine Similarity → Best Expert(s) → DAG Execution
         (MiniLM)     (vs manifests)      (scored)        (parallel)
```

Adding a new family? See [CONTRIBUTING.md](CONTRIBUTING.md).

## Research Papers

Model Garage is backed by three peer-reviewed research papers demonstrating its capabilities on real problems:

### [Blades: Compositional Capability Enhancement Through Hidden State Injection](research/)
Hidden state injection between specialized models achieves **+14.2% accuracy** on medical reasoning tasks. Establishes **7 validated principles** for capability transfer including the N-4 layer rule and same-domain synergy (+27.8%). Uses Model Garage's `extract`, `inject`, and `snapshot` modules.

### [Learned Routers Don't Learn: Expert Miscalibration in MoE Models](research/)
Per-layer expert isolation reveals that learned MoE routers show **Spearman rho = 0.069** between routing probability and expert quality. **482/896** expert-layer-domain combinations show statistically significant specialization (BH FDR, alpha=0.05), yet the router ignores it. v4.0 results with 5x more data, per-layer CIs, and random baselines. Uses Model Garage's `analyze` and `core.hooks` modules.

### [Sparse Pathways: Domain-Aware Neuron Routing for Efficient Inference](research/)
FFN neurons exhibit strong domain specialization (~50% in late layers), with **r=0.999 correlation** between model scale and specialization degree. Demonstrates **2-4x potential compute reduction** via negative neuron selection. Uses Model Garage's `analyze` and `snapshot` modules.

See [research/](research/) for the full findings, blade principles, and experiment index.

## CLI Commands

| Command | Description |
|---------|-------------|
| `garage open <model>` | Load and display model architecture card |
| `garage extract <model>` | Extract components (attention, FFN, embeddings) |
| `garage analyze <model>` | Analyze activations, entropy, sparsity |
| `garage compare <a> <b>` | Compare architectures, find compatible parts |
| `garage registry list` | List decomposed models in local registry |
| `garage registry add <model>` | Decompose and register a model |

## Project Structure

```
model-garage/
├── src/model_garage/
│   ├── cli/          # Retro-themed CLI (Typer + Rich)
│   ├── core/         # Model loading, hooks, tensor utils
│   ├── extract/      # Component extraction engine
│   ├── inject/       # Layer injection, blades, debate
│   ├── analyze/      # Activation analysis
│   ├── compose/      # Hybrid model builder
│   ├── registry/     # Component registry & decomposition
│   ├── snapshot/     # Hidden state capture
│   └── models/       # Model family adapters
├── examples/         # Runnable example scripts
├── research/         # Validated findings & principles
├── tests/            # Unit & integration tests
└── docs/             # Documentation
```

## Philosophy

> *"People used to be 'Gear Heads' — they could fine-tune actual car engines manually. They understood every part, could diagnose problems by sound, and could build a hot rod from junkyard parts.*
>
> *Modern AI models are the same kind of machine. They have parts — attention heads, feed-forward networks, embeddings, normalization layers. These parts can be extracted, tested, swapped, and recombined.*
>
> *Model Garage makes you a neural network gear head."*

## Contributing

We especially welcome contributions that add support for new model families. See [CONTRIBUTING.md](CONTRIBUTING.md) for the guide on writing a new `ModelDecomposer`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
