#!/usr/bin/env python3
"""Example: Add self-debate between model layers."""

import torch
from model_garage.core.loader import quick_load
from model_garage.inject.debate import SelfDebate

model, tokenizer, _ = quick_load("gpt2")
device = next(model.parameters()).device
input_ids = tokenizer("Artificial intelligence will", return_tensors="pt").input_ids.to(device)

debate = SelfDebate(model, layer_idx=6, divergence_method="dropout", divergence_strength=0.15)

with debate:
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=30, do_sample=True, temperature=0.8)
    info = debate.get_debate_info()

print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
if info:
    print(f"Cosine sim between perspectives: {info[0]['cosine_similarity']:.4f}")
