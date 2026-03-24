"""
Model Garage - Open the hood on neural networks.

Component-level model surgery, analysis, and composition toolkit.
Extract, analyze, inject, and compose transformer model parts.
"""

__version__ = "0.1.0"

from model_garage.core.loader import ModelLoader, quick_load
from model_garage.core.hooks import HookManager
from model_garage.core.tensor import TensorUtils, Projector
from model_garage.registry.models import ModelRegistry, ModelFamily, PartType

__all__ = [
    "ModelLoader",
    "quick_load",
    "HookManager",
    "TensorUtils",
    "Projector",
    "ModelRegistry",
    "ModelFamily",
    "PartType",
]
