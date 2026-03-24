"""Tests for model_garage.core modules."""

import pytest
import torch
import torch.nn as nn

from model_garage.core.loader import ModelLoader, quick_load
from model_garage.core.hooks import HookManager, HookHandle
from model_garage.core.tensor import TensorUtils, Projector
from model_garage.core.device import get_device, DeviceManager


class TestModelLoader:
    def test_init_auto_device(self):
        loader = ModelLoader()
        assert loader.device in ("cuda", "cpu")

    def test_init_explicit_device(self):
        loader = ModelLoader(device="cpu")
        assert loader.device == "cpu"

    def test_load_gpt2(self, gpt2_model):
        model, tokenizer, info = gpt2_model
        assert model is not None
        assert tokenizer is not None
        assert info["model_id"] == "gpt2"
        assert info["hidden_size"] == 768
        assert info["num_layers"] == 12
        assert info["vocab_size"] == 50257

    def test_get_layer_names_gpt2(self, gpt2_model):
        model, _, _ = gpt2_model
        loader = ModelLoader(device="cpu")
        names = loader.get_layer_names(model)
        assert "embedding" in names
        assert "layers" in names
        assert len(names["layers"]) == 12
        assert names["layers"][0] == "transformer.h.0"

    def test_quick_load(self):
        model, tokenizer, info = quick_load("gpt2", device="cpu")
        assert model is not None
        assert info["architecture"] == "GPT2LMHeadModel"


class TestHookManager:
    def test_init(self, gpt2_model):
        model, _, _ = gpt2_model
        mgr = HookManager(model)
        assert len(mgr.hooks) == 0

    def test_register_forward_hook(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        mgr = HookManager(model)
        name = mgr.register_forward_hook("transformer.h.0", lambda m, i, o: None)
        assert name in mgr.hooks
        mgr.remove_all()

    def test_capture_hook(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        mgr = HookManager(model)
        mgr.register_capture_hook("transformer.h.0", hook_name="layer0")

        with torch.no_grad():
            model(**sample_input)

        data = mgr.get_captured("layer0")
        assert data is not None
        assert "output" in data
        assert data["output"].shape[-1] == 768  # hidden dim
        mgr.remove_all()

    def test_injection_hook(self, gpt2_model, sample_input):
        model, _, _ = gpt2_model
        mgr = HookManager(model)

        scale_factor = 0.5
        mgr.register_injection_hook(
            "transformer.h.0",
            lambda x: x * scale_factor,
            hook_name="scale_test",
        )

        with torch.no_grad():
            output = model(**sample_input)

        assert output is not None
        mgr.remove_all()

    def test_context_manager(self, gpt2_model):
        model, _, _ = gpt2_model
        with HookManager(model) as mgr:
            mgr.register_capture_hook("transformer.h.0", hook_name="test")
            assert len(mgr.hooks) == 1
        # Hooks should be cleaned up
        assert len(mgr.hooks) == 0

    def test_list_hooks(self, gpt2_model):
        model, _, _ = gpt2_model
        mgr = HookManager(model)
        mgr.register_capture_hook("transformer.h.0", hook_name="a")
        mgr.register_capture_hook("transformer.h.1", hook_name="b")
        assert set(mgr.list_hooks()) == {"a", "b"}
        mgr.remove_all()

    def test_remove_specific_hook(self, gpt2_model):
        model, _, _ = gpt2_model
        mgr = HookManager(model)
        mgr.register_capture_hook("transformer.h.0", hook_name="keep")
        mgr.register_capture_hook("transformer.h.1", hook_name="remove")
        mgr.remove_hook("remove")
        assert "remove" not in mgr.hooks
        assert "keep" in mgr.hooks
        mgr.remove_all()


class TestTensorUtils:
    def test_ensure_device(self):
        t = torch.randn(3, 3)
        result = TensorUtils.ensure_device(t, "cpu")
        assert result.device == torch.device("cpu")

    def test_cosine_similarity_identical(self):
        t = torch.randn(10)
        sim = TensorUtils.cosine_similarity(t, t)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        sim = TensorUtils.cosine_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_l2_distance_zero(self):
        t = torch.randn(10)
        assert TensorUtils.l2_distance(t, t) < 1e-5

    def test_l2_distance_nonzero(self):
        a = torch.zeros(10)
        b = torch.ones(10)
        dist = TensorUtils.l2_distance(a, b)
        assert dist > 0

    def test_stats(self):
        t = torch.randn(100)
        stats = TensorUtils.stats(t)
        assert "shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "sparsity" in stats
        assert stats["shape"] == [100]

    def test_ensure_shape_add_batch(self):
        t = torch.randn(10, 768)
        result = TensorUtils.ensure_shape(t, (1, 10, 768))
        assert result.shape == (1, 10, 768)

    def test_ensure_shape_pad_seq(self):
        t = torch.randn(1, 5, 768)
        result = TensorUtils.ensure_shape(t, (1, 10, 768))
        assert result.shape == (1, 10, 768)

    def test_ensure_shape_truncate_seq(self):
        t = torch.randn(1, 20, 768)
        result = TensorUtils.ensure_shape(t, (1, 10, 768))
        assert result.shape == (1, 10, 768)


class TestProjector:
    def test_forward(self):
        proj = Projector(768, 256, device="cpu")
        t = torch.randn(1, 10, 768)
        result = proj.forward(t)
        assert result.shape == (1, 10, 256)

    def test_save_load(self, tmp_path):
        proj = Projector(768, 256, device="cpu")
        path = str(tmp_path / "proj.pt")
        proj.save(path)
        loaded = Projector.load(path, device="cpu")
        assert loaded.from_dim == 768
        assert loaded.to_dim == 256


class TestDeviceManager:
    def test_auto_device(self):
        dm = DeviceManager()
        assert dm.device.type in ("cuda", "cpu")

    def test_explicit_cpu(self):
        dm = DeviceManager("cpu")
        assert dm.device == torch.device("cpu")
        assert not dm.is_gpu

    def test_to_tensor(self):
        dm = DeviceManager("cpu")
        t = torch.randn(3)
        result = dm.to(t)
        assert result.device == torch.device("cpu")

    def test_to_dict(self):
        dm = DeviceManager("cpu")
        d = {"a": torch.randn(3), "b": "not_a_tensor"}
        result = dm.to_dict(d)
        assert result["a"].device == torch.device("cpu")
        assert result["b"] == "not_a_tensor"
