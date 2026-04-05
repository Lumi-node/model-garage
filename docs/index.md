---
hide:
  - navigation
  - toc
---

<style>
  .md-typeset h1 { display: none; }
</style>

<div class="hero" markdown>

<figure markdown="span">
  ![Model Garage](assets/images/hero_garage_floor.jpg){ width="900" loading=lazy }
</figure>

# **Model Garage**

### Open the hood on neural networks.

Component-level model surgery, analysis, and composition.
Extract attention heads, swap FFN layers between models, inject capability blades,
and build hybrid architectures — all from a beautiful CLI or Python API.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Lumi-node/model-garage){ .md-button }

</div>

---

<div class="grid cards" markdown>

-   :wrench:{ .lg .middle } **Extract**

    ---

    Pull real `nn.Module` components from any supported transformer.
    Attention heads, FFN layers, embeddings — ready to test in isolation.

    [:octicons-arrow-right-24: Extraction guide](guides/extraction.md)

-   :syringe:{ .lg .middle } **Inject**

    ---

    Insert custom processing between any two layers.
    Scale activations, inject adapters, add capability blades — without modifying the model.

    [:octicons-arrow-right-24: Injection guide](guides/injection.md)

-   :mag:{ .lg .middle } **Analyze**

    ---

    Capture hidden states, measure entropy, sparsity, and activation patterns.
    Understand what each layer actually does.

    [:octicons-arrow-right-24: Analysis guide](guides/analysis.md)

-   :jigsaw:{ .lg .middle } **Compose**

    ---

    Register models, compare architectures, and find compatible parts.
    Build hybrid models from proven components.

    [:octicons-arrow-right-24: Composition guide](guides/composition.md)

</div>

---

## Quick Example

=== "CLI"

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

=== "Python"

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

---

## Supported Architectures

70+ models validated across 18 vendors — but Model Garage works on **any PyTorch transformer**. If a model follows standard attention/FFN/norm patterns, the existing decomposers will detect it automatically. Adding a new family is a single class. See [Contributing](contributing.md).

| Family | Models | Capabilities |
|--------|--------|-------------|
| **GPT-2** | gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2 | Extract, inject, analyze, compose |
| **Llama** | Llama-2-7b, Llama-3-8b, TinyLlama, CodeLlama | Extract, inject, analyze, compose |
| **Phi** | Phi-2, Phi-3.5, Phi-4, Phi-4-reasoning, MediPhi | Extract, inject, analyze, compose |
| **Phi-MoE** | Phi-3.5-MoE, Phi-mini-MoE, Phi-tiny-MoE | Extract, inject, MoE routing, blade injection |
| **Mistral** | Mistral-7B, Mixtral-8x7B | Extract, inject, analyze, compose |
| **Gemma** | Gemma-2b/7b, Gemma-3, FunctionGemma, MedGemma | Extract, inject, analyze, compose |
| **Qwen** | Qwen-1.5, Qwen-2, Kimi-K2.5 | Extract, inject, analyze, compose |
| **BERT** | bert-base/large, distilbert, MiniLM, mpnet | Extraction, analysis |
| **Protein** | ESM2 (8M to 3B) | Extraction, analysis |
| **BitNet** | bitnet-b1.58-2B-4T | Extraction, analysis |

---

## Backed by Research

Model Garage is validated through three peer-reviewed papers with quantified results.

<div class="grid cards" markdown>

-   **Blades: Capability Transfer**

    ---

    Hidden state injection achieves **+14.2% accuracy** on medical reasoning.
    7 validated principles for capability transfer.

    [:octicons-arrow-right-24: Read more](research/blades.md)

-   **MoE Router Miscalibration**

    ---

    Learned MoE routers show **rho = 0.069** between routing probability and expert quality.
    482/896 combinations show significant specialization — yet routers ignore it.

    [:octicons-arrow-right-24: Read more](research/moe-routing.md)

-   **Sparse Pathways**

    ---

    FFN neurons show domain specialization with **r=0.999** scale correlation.
    **2-4x potential compute reduction** via negative neuron selection.

    [:octicons-arrow-right-24: Read more](research/sparse-pathways.md)

</div>
