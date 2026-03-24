"""Shared fixtures for Model Garage tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def gpt2_model():
    """Load GPT-2 (small) once for all tests."""
    from model_garage.core.loader import ModelLoader

    loader = ModelLoader(device="cpu")
    model, tokenizer, info = loader.load("gpt2", dtype=torch.float32)
    return model, tokenizer, info


@pytest.fixture(scope="session")
def gpt2_parts(gpt2_model):
    """Unpack GPT-2 fixture."""
    return gpt2_model


@pytest.fixture
def sample_input(gpt2_model):
    """Create a sample input for GPT-2."""
    _, tokenizer, _ = gpt2_model
    return tokenizer("The quick brown fox", return_tensors="pt")


@pytest.fixture
def hidden_states():
    """Create random hidden states matching GPT-2 dimensions."""
    return torch.randn(1, 10, 768)  # batch=1, seq=10, hidden=768
