#!/usr/bin/env python3
"""Example: Inject custom processing between model layers."""

import torch
from model_garage.core.loader import quick_load
from model_garage.inject.layer import LayerInjector

model, tokenizer, info = quick_load("gpt2")
device = next(model.parameters()).device
input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to(device)

# Baseline
with torch.no_grad():
    baseline = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print("Baseline:", tokenizer.decode(baseline[0], skip_special_tokens=True))

# Scale activations at layer 6
with LayerInjector(model) as injector:
    injector.inject_scaling("transformer.h.6", scale=0.9)
    with torch.no_grad():
        modified = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print("Scaled:  ", tokenizer.decode(modified[0], skip_special_tokens=True))
