"""Integration test: full extract -> inject -> analyze flow."""

import pytest
import torch

from model_garage.core.loader import quick_load
from model_garage.core.hooks import HookManager
from model_garage.core.tensor import TensorUtils
from model_garage.extract.pytorch import PyTorchExtractor, ComponentTester
from model_garage.inject.layer import LayerInjector
from model_garage.inject.debate import SelfDebate
from model_garage.registry.models import ModelRegistry
from model_garage.snapshot.capture import SnapshotCapture, LayerSnapshot


class TestFullFlow:
    """Test the complete Model Garage workflow."""

    def test_load_decompose_extract(self, gpt2_model):
        """Load a model, decompose it, extract a component."""
        model, tokenizer, info = gpt2_model

        # Decompose
        registry = ModelRegistry()
        spec = registry.register("gpt2", model)
        assert len(spec.parts) > 50  # GPT-2 has ~64 parts

        # Extract via PyTorchExtractor
        extractor = PyTorchExtractor("gpt2", device="cpu")
        extractor.model = model
        extractor.config = model.config
        extractor.arch_type = "gpt2"
        extractor.arch_pattern = extractor.ARCH_PATTERNS["gpt2"]

        attn = extractor.extract_component("self_attention", layer_idx=6)
        assert attn.spec.input_dim == 768

        # Test extracted component in isolation
        tester = ComponentTester(device="cpu")
        result = tester.test_attention(attn, seq_len=16)
        assert result["success"]

    def test_inject_and_verify(self, gpt2_model, sample_input):
        """Inject modification and verify output changes."""
        model, _, _ = gpt2_model

        with torch.no_grad():
            baseline = model(**sample_input).logits.clone()

        # Inject scaling at layer 6
        with LayerInjector(model) as injector:
            injector.inject_scaling("transformer.h.6", scale=0.1)
            with torch.no_grad():
                modified = model(**sample_input).logits

        # Output should be different
        assert not torch.allclose(baseline, modified, atol=1e-3)

    def test_capture_snapshots(self, gpt2_model, sample_input):
        """Capture hidden state snapshots across layers."""
        model, _, _ = gpt2_model

        capture = SnapshotCapture(model)
        snapshots = capture.run(
            sample_input["input_ids"],
            layers=["transformer.h.0", "transformer.h.6", "transformer.h.11"],
        )

        assert len(snapshots) == 3
        for name, snap in snapshots.items():
            assert isinstance(snap, LayerSnapshot)
            assert snap.shape[-1] == 768
            assert 0 <= snap.sparsity <= 1

    def test_analyze_activations(self, gpt2_model, sample_input):
        """Analyze activations across all layers."""
        model, _, _ = gpt2_model

        mgr = HookManager(model)
        layers = [f"transformer.h.{i}" for i in range(12)]
        for ln in layers:
            mgr.register_capture_hook(ln, hook_name=ln)

        with torch.no_grad():
            model(**sample_input)

        for ln in layers:
            data = mgr.get_captured(ln)
            assert data is not None
            stats = TensorUtils.stats(data["output"])
            assert "mean" in stats
            assert "sparsity" in stats

        mgr.remove_all()

    def test_debate_generates_output(self, gpt2_model):
        """Debate chamber produces valid generation."""
        model, tokenizer, _ = gpt2_model
        input_ids = tokenizer("Hello", return_tensors="pt").input_ids

        with SelfDebate(model, layer_idx=6, divergence_strength=0.1):
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=10, do_sample=False)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        assert len(decoded) > 5  # Generated something
