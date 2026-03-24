"""Tests for model_garage.registry.models."""

import pytest
import torch

from model_garage.registry.models import (
    ModelRegistry,
    ModelFamily,
    PartType,
    PartSpec,
    ModelSpec,
    GPT2Decomposer,
    LlamaDecomposer,
    PhiDecomposer,
)


class TestModelFamily:
    def test_enum_values(self):
        assert ModelFamily.GPT2.value == "gpt2"
        assert ModelFamily.LLAMA.value == "llama"
        assert ModelFamily.PHI.value == "phi"


class TestPartType:
    def test_enum_values(self):
        assert PartType.ATTENTION.value == "attention"
        assert PartType.FFN.value == "ffn"
        assert PartType.EMBEDDING.value == "embedding"
        assert PartType.FULL_LAYER.value == "full_layer"


class TestGPT2Decomposer:
    def test_detect(self, gpt2_model):
        model, _, _ = gpt2_model
        decomposer = GPT2Decomposer()
        assert decomposer.detect(model, "gpt2")

    def test_decompose(self, gpt2_model):
        model, _, _ = gpt2_model
        decomposer = GPT2Decomposer()
        spec = decomposer.decompose(model, "gpt2")

        assert spec.family == ModelFamily.GPT2
        assert spec.hidden_dim == 768
        assert spec.num_layers == 12
        assert spec.num_heads == 12

        # Check parts exist
        assert "embedding" in spec.parts
        assert "attention_0" in spec.parts
        assert "ffn_0" in spec.parts
        assert "layer_0" in spec.parts
        assert "output_head" in spec.parts

    def test_get_attention_part(self, gpt2_model):
        model, _, _ = gpt2_model
        decomposer = GPT2Decomposer()
        spec = decomposer.decompose(model, "gpt2")

        attn = spec.get_attention(0)
        assert attn is not None
        assert attn.part_type == PartType.ATTENTION
        assert attn.input_dim == 768
        assert attn.num_heads == 12

    def test_get_ffn_part(self, gpt2_model):
        model, _, _ = gpt2_model
        decomposer = GPT2Decomposer()
        spec = decomposer.decompose(model, "gpt2")

        ffn = spec.get_ffn(5)
        assert ffn is not None
        assert ffn.part_type == PartType.FFN
        assert ffn.intermediate_dim == 768 * 4

    def test_get_module(self, gpt2_model):
        model, _, _ = gpt2_model
        decomposer = GPT2Decomposer()
        spec = decomposer.decompose(model, "gpt2")

        part = spec.parts["attention_0"]
        module = decomposer.get_module(model, part)
        assert module is not None
        assert hasattr(module, "forward")


class TestModelRegistry:
    def test_register_gpt2(self, gpt2_model):
        model, _, _ = gpt2_model
        registry = ModelRegistry()
        spec = registry.register("gpt2", model)

        assert spec.model_id == "gpt2"
        assert spec.family == ModelFamily.GPT2
        assert len(spec.parts) > 0

    def test_list_models(self, gpt2_model):
        model, _, _ = gpt2_model
        registry = ModelRegistry()
        registry.register("gpt2", model)
        assert "gpt2" in registry.list_models()

    def test_get_spec(self, gpt2_model):
        model, _, _ = gpt2_model
        registry = ModelRegistry()
        registry.register("gpt2", model)

        spec = registry.get_spec("gpt2")
        assert spec is not None
        assert spec.hidden_dim == 768

    def test_get_part(self, gpt2_model):
        model, _, _ = gpt2_model
        registry = ModelRegistry()
        registry.register("gpt2", model)

        part = registry.get_part("gpt2", "attention_0")
        assert part is not None
        assert part.part_type == PartType.ATTENTION

    def test_list_parts_by_type(self, gpt2_model):
        model, _, _ = gpt2_model
        registry = ModelRegistry()
        registry.register("gpt2", model)

        attn_parts = registry.list_parts("gpt2", PartType.ATTENTION)
        assert len(attn_parts) == 12  # 12 layers

    def test_cache_save_load(self, gpt2_model, tmp_path):
        model, _, _ = gpt2_model
        registry = ModelRegistry(cache_dir=tmp_path)
        registry.register("gpt2", model)

        # Create a new registry that loads from cache
        registry2 = ModelRegistry(cache_dir=tmp_path)
        spec = registry2.get_spec("gpt2")
        assert spec is not None
        assert spec.hidden_dim == 768

    def test_detect_family(self, gpt2_model):
        model, _, _ = gpt2_model
        registry = ModelRegistry()
        family = registry.detect_family(model, "gpt2")
        assert family == ModelFamily.GPT2
