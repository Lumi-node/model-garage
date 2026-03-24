#!/usr/bin/env python3
"""Example: Open a model and inspect its architecture."""

from model_garage.core.loader import ModelLoader

loader = ModelLoader()
model, tokenizer, info = loader.load("gpt2")

print("Model Info:")
for key, value in info.items():
    print(f"  {key}: {value}")

print("\nLayer Names:")
layer_names = loader.get_layer_names(model)
for key, value in layer_names.items():
    if isinstance(value, list):
        print(f"  {key}: {value[0]} ... {value[-1]} ({len(value)} layers)")
    else:
        print(f"  {key}: {value}")
