"""Tests for model_garage.extract modules."""

import pytest
import torch

from model_garage.extract.pytorch import PyTorchExtractor, ComponentTester, ExtractedComponent


class TestPyTorchExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        ext = PyTorchExtractor("gpt2", device="cpu")
        ext.load_model()
        return ext

    def test_load_model(self, extractor):
        assert extractor.model is not None
        assert extractor.arch_type == "gpt2"
        assert extractor.config is not None

    def test_detect_architecture(self, extractor):
        assert extractor.arch_type == "gpt2"

    def test_get_num_layers(self, extractor):
        assert extractor.get_num_layers() == 12

    def test_get_hidden_size(self, extractor):
        assert extractor.get_hidden_size() == 768

    def test_summary(self, extractor):
        s = extractor.summary()
        assert s["model_name"] == "gpt2"
        assert s["architecture"] == "gpt2"
        assert s["hidden_size"] == 768
        assert s["num_layers"] == 12

    def test_list_components(self, extractor):
        components = extractor.list_available_components()
        assert len(components) > 0
        assert "self_attention.0" in components
        assert "feed_forward.0" in components
        assert "embeddings" in components

    def test_extract_attention(self, extractor):
        attn = extractor.extract_component("self_attention", layer_idx=0)
        assert isinstance(attn, ExtractedComponent)
        assert attn.spec.component_type == "attention"
        assert attn.spec.input_dim == 768
        assert attn.source_model == "gpt2"
        assert attn.source_layer == 0

    def test_extract_ffn(self, extractor):
        ffn = extractor.extract_component("feed_forward", layer_idx=0)
        assert isinstance(ffn, ExtractedComponent)
        assert ffn.spec.component_type == "ffn"

    def test_extract_layer(self, extractor):
        components = extractor.extract_layer(0)
        assert "self_attention" in components
        assert "feed_forward" in components

    def test_extract_embeddings(self, extractor):
        emb = extractor.extract_component("embeddings")
        assert emb.spec.component_type == "embedding"

    def test_extracted_component_has_parameters(self, extractor):
        attn = extractor.extract_component("self_attention", layer_idx=0)
        params = list(attn.parameters())
        assert len(params) > 0

    def test_caching(self, extractor):
        a1 = extractor.extract_component("self_attention", layer_idx=0)
        a2 = extractor.extract_component("self_attention", layer_idx=0)
        assert a1 is a2  # Same object from cache


class TestComponentTester:
    def test_test_attention(self, gpt2_model):
        model, _, _ = gpt2_model
        extractor = PyTorchExtractor("gpt2", device="cpu")
        extractor.model = model
        extractor.config = model.config
        extractor.arch_type = "gpt2"
        extractor.arch_pattern = extractor.ARCH_PATTERNS["gpt2"]

        attn = extractor.extract_component("self_attention", layer_idx=0)
        tester = ComponentTester(device="cpu")
        result = tester.test_attention(attn, seq_len=32)
        assert result["success"]
        assert result["output_shape"][0] == 1  # batch
        assert result["output_shape"][-1] == 768  # hidden

    def test_test_ffn(self, gpt2_model):
        model, _, _ = gpt2_model
        extractor = PyTorchExtractor("gpt2", device="cpu")
        extractor.model = model
        extractor.config = model.config
        extractor.arch_type = "gpt2"
        extractor.arch_pattern = extractor.ARCH_PATTERNS["gpt2"]

        ffn = extractor.extract_component("feed_forward", layer_idx=0)
        tester = ComponentTester(device="cpu")
        result = tester.test_ffn(ffn, seq_len=32)
        assert result["success"]
