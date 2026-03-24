"""
Snapshot capture - capture hidden states mid-inference.

Like a high-speed camera on the dyno - freeze the action at any point.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from model_garage.core.hooks import HookManager


@dataclass
class LayerSnapshot:
    """A snapshot of hidden states at a specific layer."""
    layer_name: str
    hidden_states: torch.Tensor
    mean_activation: float
    std_activation: float
    sparsity: float
    shape: list
    dtype: str

    @classmethod
    def from_tensor(cls, layer_name: str, tensor: torch.Tensor) -> "LayerSnapshot":
        """Create a snapshot from a captured tensor."""
        return cls(
            layer_name=layer_name,
            hidden_states=tensor.detach().cpu(),
            mean_activation=tensor.float().mean().item(),
            std_activation=tensor.float().std().item(),
            sparsity=(tensor == 0).float().mean().item(),
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
        )


class SnapshotCapture:
    """
    Capture hidden state snapshots during model inference.

    Usage:
        capture = SnapshotCapture(model)
        snapshots = capture.run(input_ids, layers=["model.layers.0", "model.layers.15"])
        for name, snap in snapshots.items():
            print(f"{name}: mean={snap.mean_activation:.4f}, sparsity={snap.sparsity:.2%}")
    """

    # Common layer patterns for auto-detection
    LAYER_PATTERNS = {
        "llama": "model.layers.{i}",
        "gpt2": "transformer.h.{i}",
        "bert": "encoder.layer.{i}",
        "t5": "decoder.block.{i}",
    }

    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_manager = HookManager(model)

    def run(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[str]] = None,
        num_layers: Optional[int] = None,
    ) -> Dict[str, LayerSnapshot]:
        """
        Run inference and capture snapshots.

        Args:
            input_ids: Input token IDs
            layers: Specific layer names to capture. If None, auto-detect.
            num_layers: Number of layers to capture (evenly spaced). Used with auto-detect.

        Returns:
            Dict mapping layer names to LayerSnapshot objects
        """
        if layers is None:
            layers = self._auto_detect_layers(num_layers)

        for ln in layers:
            self.hook_manager.register_capture_hook(ln, hook_name=ln)

        with torch.no_grad():
            self.model(input_ids)

        snapshots = {}
        for ln in layers:
            data = self.hook_manager.get_captured(ln)
            if data and "output" in data:
                snapshots[ln] = LayerSnapshot.from_tensor(ln, data["output"])

        self.hook_manager.remove_all()
        return snapshots

    def _auto_detect_layers(self, num_layers: Optional[int] = None) -> List[str]:
        """Auto-detect layer names from model structure."""
        for name, module in self.model.named_modules():
            parts = name.split(".")
            if len(parts) >= 2 and parts[-1].isdigit():
                parent = ".".join(parts[:-1])
                parent_module = dict(self.model.named_modules()).get(parent)
                if parent_module and isinstance(parent_module, nn.ModuleList):
                    total = len(parent_module)
                    if num_layers and num_layers < total:
                        step = total // num_layers
                        indices = list(range(0, total, step))[:num_layers]
                    else:
                        indices = list(range(total))
                    return [f"{parent}.{i}" for i in indices]

        return []
