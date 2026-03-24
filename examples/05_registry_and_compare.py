#!/usr/bin/env python3
"""Example: Register models and compare architectures."""

from model_garage.core.loader import ModelLoader
from model_garage.registry.models import ModelRegistry

loader = ModelLoader()
registry = ModelRegistry()

print("Loading gpt2...")
model_a, _, _ = loader.load("gpt2")
spec_a = registry.register("gpt2", model_a)
print(f"  {spec_a.family.value}, {spec_a.hidden_dim}d, {spec_a.num_layers}L, {len(spec_a.parts)} parts")

print("Loading distilgpt2...")
model_b, _, _ = loader.load("distilgpt2")
spec_b = registry.register("distilgpt2", model_b)
print(f"  {spec_b.family.value}, {spec_b.hidden_dim}d, {spec_b.num_layers}L, {len(spec_b.parts)} parts")

comparison = registry.compare("gpt2", "distilgpt2")
print(f"\nHidden dims: {comparison['hidden_dims']}")
print(f"Layers: {comparison['num_layers']}")
print(f"Compatible: {comparison['compatible_parts']}")
