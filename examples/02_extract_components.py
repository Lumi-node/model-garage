#!/usr/bin/env python3
"""Example: Extract and test individual model components."""

from model_garage.extract.pytorch import PyTorchExtractor, ComponentTester

extractor = PyTorchExtractor("gpt2")
extractor.load_model()

print(f"Model: {extractor.model_name}")
print(f"Architecture: {extractor.arch_type}")
print(f"Layers: {extractor.get_num_layers()}")
print(f"Hidden size: {extractor.get_hidden_size()}")

# Extract attention from layer 6
attention = extractor.extract_component("self_attention", layer_idx=6)
print(f"\nExtracted: {attention.spec.name}")
print(f"  Input dim: {attention.spec.input_dim}")
print(f"  Heads: {attention.spec.num_heads}")
print(f"  Parameters: {sum(p.numel() for p in attention.parameters()):,}")

# Test it
tester = ComponentTester()
result = tester.test_attention(attention)
print(f"\nTest: {'PASS' if result['success'] else 'FAIL'}")
if result["success"]:
    print(f"  Output shape: {result['output_shape']}")
