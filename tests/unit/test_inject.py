"""Tests for model_garage.inject modules."""

import pytest
import torch

from model_garage.inject.layer import LayerInjector
from model_garage.inject.debate import DebateChamber, SelfDebate
from model_garage.inject.temperature import (
    RandomSwitchDebate,
    FilteredBlendDebate,
    AdaptiveDebate,
    debate_sample,
)


class TestLayerInjector:
    def test_inject_identity(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model

        # Get baseline output
        with torch.no_grad():
            baseline = model(**sample_input).logits

        # Inject identity — should not change output
        injector = LayerInjector(model)
        injector.inject_identity("transformer.h.6")
        with torch.no_grad():
            modified = model(**sample_input).logits
        injector.remove_all()

        assert torch.allclose(baseline, modified, atol=1e-5)

    def test_inject_scaling(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        injector = LayerInjector(model)
        name = injector.inject_scaling("transformer.h.6", scale=0.5)
        assert name in injector.list_injections()

        with torch.no_grad():
            model(**sample_input)

        injector.remove_all()
        assert len(injector.list_injections()) == 0

    def test_inject_noise(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        injector = LayerInjector(model)
        injector.inject_noise("transformer.h.0", noise_scale=0.01)

        with torch.no_grad():
            model(**sample_input)

        injector.remove_all()

    def test_context_manager(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        with LayerInjector(model) as injector:
            injector.inject_scaling("transformer.h.0", scale=0.9)
            with torch.no_grad():
                model(**sample_input)

    def test_remove_specific(self, gpt2_model):
        model, _, _ = gpt2_model
        injector = LayerInjector(model)
        n1 = injector.inject_identity("transformer.h.0")
        n2 = injector.inject_identity("transformer.h.1")
        injector.remove(n1)
        assert n1 not in injector.list_injections()
        assert n2 in injector.list_injections()
        injector.remove_all()


class TestDebateChamber:
    def test_dropout_debate(self, hidden_states):
        chamber = DebateChamber(768, divergence_method="dropout")
        output = chamber(hidden_states)
        assert output.shape == hidden_states.shape

    def test_perturbation_debate(self, hidden_states):
        chamber = DebateChamber(768, divergence_method="perturbation")
        output = chamber(hidden_states)
        assert output.shape == hidden_states.shape

    def test_projection_debate(self, hidden_states):
        chamber = DebateChamber(768, divergence_method="projection")
        output = chamber(hidden_states)
        assert output.shape == hidden_states.shape

    def test_gated_reconciliation(self, hidden_states):
        chamber = DebateChamber(768, reconciliation_method="gated")
        output = chamber(hidden_states)
        assert output.shape == hidden_states.shape

    def test_confidence_reconciliation(self, hidden_states):
        chamber = DebateChamber(768, reconciliation_method="confidence")
        output = chamber(hidden_states)
        assert output.shape == hidden_states.shape

    def test_forward_with_info(self, hidden_states):
        chamber = DebateChamber(768)
        output, info = chamber.forward_with_info(hidden_states)
        assert output.shape == hidden_states.shape
        assert "cosine_similarity" in info
        assert "l2_difference" in info


class TestSelfDebate:
    def test_enable_disable(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        debate = SelfDebate(model, layer_idx=6)
        debate.enable()
        with torch.no_grad():
            model(**sample_input)
        info = debate.get_debate_info()
        assert len(info) > 0
        debate.disable()

    def test_context_manager(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        with SelfDebate(model, layer_idx=6) as debate:
            with torch.no_grad():
                model(**sample_input)


class TestTemperatureDebate:
    def test_random_switch(self):
        logits = torch.randn(1, 50257)
        debate = RandomSwitchDebate()
        probs, info = debate.debate(logits)
        assert probs.shape == logits.shape
        assert info["strategy"] == "random_switch"

    def test_filtered_blend(self):
        logits = torch.randn(1, 50257)
        debate = FilteredBlendDebate()
        probs, info = debate.debate(logits)
        assert probs.shape == logits.shape
        assert info["strategy"] == "filtered_blend"

    def test_adaptive(self):
        logits = torch.randn(1, 50257)
        debate = AdaptiveDebate()
        probs, info = debate.debate(logits)
        assert probs.shape == logits.shape
        assert info["strategy"] == "adaptive"

    def test_debate_sample(self):
        logits = torch.randn(1, 50257)
        tokens = debate_sample(logits, strategy="random_switch")
        assert tokens.shape == (1, 1)
