"""
Activation analyzer for PyTorch models.

Like reading the OBD-II codes - understand what each layer is doing.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from model_garage.core.hooks import HookManager
from model_garage.core.tensor import TensorUtils

logger = logging.getLogger(__name__)


class ActivationAnalyzer:
    """
    Analyze model activations layer by layer.

    Uses hooks to capture hidden states during inference and compute
    statistics about activation patterns.
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or str(next(model.parameters()).device)
        self.hook_manager = HookManager(model)
        self._results = {}

    def analyze_layer(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Analyze activations at a specific layer.

        Args:
            layer_name: Layer to analyze (e.g., "model.layers.12")
            input_ids: Input token IDs

        Returns:
            Dict with activation statistics
        """
        self.hook_manager.register_capture_hook(layer_name, hook_name=layer_name)

        with torch.no_grad():
            self.model(input_ids.to(self.device))

        data = self.hook_manager.get_captured(layer_name)
        self.hook_manager.remove_hook(layer_name)

        if data is None or "output" not in data:
            return {"error": f"No data captured for {layer_name}"}

        stats = TensorUtils.stats(data["output"])
        stats["layer"] = layer_name
        return stats

    def analyze_all_layers(
        self,
        layer_names: list,
        input_ids: torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze activations across multiple layers.

        Args:
            layer_names: List of layer names
            input_ids: Input token IDs

        Returns:
            Dict mapping layer names to their stats
        """
        for ln in layer_names:
            self.hook_manager.register_capture_hook(ln, hook_name=ln)

        with torch.no_grad():
            self.model(input_ids.to(self.device))

        results = {}
        for ln in layer_names:
            data = self.hook_manager.get_captured(ln)
            if data and "output" in data:
                results[ln] = TensorUtils.stats(data["output"])
                results[ln]["layer"] = ln

        self.hook_manager.remove_all()
        self._results = results
        return results

    def compute_entropy(self, layer_name: str, input_ids: torch.Tensor) -> float:
        """Compute entropy of activation distribution at a layer."""
        self.hook_manager.register_capture_hook(layer_name, hook_name=layer_name)

        with torch.no_grad():
            self.model(input_ids.to(self.device))

        data = self.hook_manager.get_captured(layer_name)
        self.hook_manager.remove_hook(layer_name)

        if data is None or "output" not in data:
            return float("nan")

        output = data["output"].float()
        probs = torch.softmax(output, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        return entropy

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        """Get last analysis results."""
        return self._results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hook_manager.remove_all()
