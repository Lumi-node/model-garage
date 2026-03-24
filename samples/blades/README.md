# Blade Examples

A **blade** is a hidden state injection that transfers a learned capability from a source model to a target model.

## How Blades Work

```
Source Model (specialist)        Target Model (generalist)
┌─────────────────────┐         ┌─────────────────────┐
│  Layer 0            │         │  Layer 0            │
│  Layer 1            │         │  Layer 1            │
│  ...                │         │  ...                │
│  Layer N-4 ─────────┼────┐    │  Layer N-4  ◄───────┼──── Gated injection
│  ...                │    │    │  ...                │
│  Layer N            │    │    │  Layer N            │
└─────────────────────┘    │    └─────────────────────┘
                           │
                    Hidden state captured
                    at 87.5% depth (N-4)
```

## Creating a Blade

```python
from model_garage.core.loader import quick_load
from model_garage.core.hooks import HookManager
from model_garage.core.serialization import save_component, ComponentMetadata
import torch

# Load specialist model
model, tokenizer, _ = quick_load("microsoft/Phi-4-mini-reasoning")
hook_mgr = HookManager(model)

# Capture hidden state at layer N-4 (the optimal injection point)
n_layers = model.config.num_hidden_layers
blade_layer = n_layers - 4
layer_name = f"model.layers.{blade_layer}"

hook_mgr.register_capture_hook(layer_name, hook_name="blade_capture")

# Run specialist on domain-specific prompt
inputs = tokenizer("Solve: What is the derivative of x^3?", return_tensors="pt")
with torch.no_grad():
    model(**inputs.to(next(model.parameters()).device))

# Save the blade
blade_data = hook_mgr.get_captured("blade_capture")
save_component(
    blade_data["output"],
    "reasoning_blade",
    metadata=ComponentMetadata(
        component_type="blade",
        source_model="microsoft/Phi-4-mini-reasoning",
        layer_index=blade_layer,
        extraction_date="2026-03-24",
        toolkit_version="0.1.0",
        hidden_dim=model.config.hidden_size,
        num_heads=None,
        compatible_with=["phi", "llama"],
        notes="Reasoning blade from layer N-4, gated injection recommended"
    )
)
hook_mgr.remove_all()
```

## Injecting a Blade

```python
from model_garage.core.loader import quick_load
from model_garage.core.serialization import load_component
from model_garage.inject.layer import LayerInjector
import torch

# Load target model
target, tokenizer, _ = quick_load("microsoft/MediPhi")

# Load blade
blade_state, metadata = load_component("reasoning_blade", return_metadata=True)
blade_tensor = blade_state["output"]  # or blade_state["tensor"]

# Create gated injection function
gate = torch.nn.Linear(target.config.hidden_size, target.config.hidden_size)
gate = gate.to(next(target.parameters()).device)

def gated_blade_injection(hidden_states):
    g = torch.sigmoid(gate(hidden_states))
    blade = blade_tensor.to(hidden_states.device)
    # Match sequence length
    if blade.shape[1] != hidden_states.shape[1]:
        blade = blade[:, :hidden_states.shape[1], :]
    return hidden_states + g * blade * 0.1  # scale factor

# Inject at N-4
n_layers = target.config.num_hidden_layers
with LayerInjector(target) as injector:
    injector.inject(f"model.layers.{n_layers - 4}", gated_blade_injection)
    inputs = tokenizer("Diagnose: patient with chest pain", return_tensors="pt")
    with torch.no_grad():
        output = target.generate(inputs.input_ids.to(next(target.parameters()).device), max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Blade Spec Format

See `example_blade_spec.json` for the metadata format.
