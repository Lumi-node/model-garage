# Samples

Example data showing what Model Garage produces.

## Directory Structure

```
samples/
├── decomposed/              # Model decomposition specs (metadata, no weights)
│   ├── gpt2_spec.json       # GPT-2: 12 layers, 768d, 64 parts
│   ├── phi35_mini_spec.json  # Phi-3.5-mini: 32 layers, 3072d, 163 parts
│   ├── phi4_reasoning_spec.json  # Phi-4-reasoning: 32 layers, 3072d
│   └── esm2_spec.json       # ESM2 protein model: 12 layers, 480d
│
├── blades/                  # Capability blade examples
│   ├── README.md            # How to create and inject blades
│   └── example_blade_spec.json  # Blade metadata format
│
└── manifests/               # Expert router manifests
    ├── medgemma-4b.json     # Medical expert manifest
    ├── lighton-ocr.json     # OCR expert manifest
    └── phi4.json            # General reasoning manifest
```

## Decomposition Specs

Each spec.json describes a fully decomposed model — every extractable part with its type, dimensions, and module path. These are metadata only (no weights). To decompose your own model:

```bash
garage registry add microsoft/phi-2
```

## Blades

See [blades/README.md](blades/README.md) for how to create and inject capability blades.

## Expert Router Manifests

JSON files that describe an expert model's capabilities, enabling zero-training semantic routing. Drop a manifest in the router's manifest directory to make a model instantly routable.
