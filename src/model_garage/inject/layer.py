"""
Layer Injector - Insert custom layers between model layers.

Like a turbocharger install - adds new capability without replacing the engine.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, List, Union
from model_garage.core.hooks import HookManager


class LayerInjector:
    """
    Inject custom processing between transformer layers.

    Proven capabilities:
    - Identity injection (pass-through)
    - Scaling injection (modify magnitude)
    - Additive injection (add signals)
    - Extraction injection (capture without modifying)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize injector for a model.

        Args:
            model: The model to inject into
        """
        self.model = model
        self.hook_manager = HookManager(model)
        self.active_injections: List[str] = []

    def inject(
        self,
        layer_name: str,
        injection_fn: Callable[[torch.Tensor], torch.Tensor],
        name: Optional[str] = None
    ) -> str:
        """
        Inject a function after a layer.

        Args:
            layer_name: Layer to inject after (e.g., "transformer.h.6")
            injection_fn: Function(hidden_states) -> modified_hidden_states
            name: Optional name for this injection

        Returns:
            Injection name for later removal
        """
        name = name or f"injection_{len(self.active_injections)}"

        hook_name = self.hook_manager.register_injection_hook(
            layer_name=layer_name,
            injection_fn=injection_fn,
            hook_name=name
        )

        self.active_injections.append(hook_name)
        return hook_name

    def inject_identity(self, layer_name: str) -> str:
        """
        Inject identity function (for testing).

        This should have NO effect on output.
        """
        return self.inject(
            layer_name=layer_name,
            injection_fn=lambda x: x,
            name=f"{layer_name}_identity"
        )

    def inject_scaling(self, layer_name: str, scale: float = 0.9) -> str:
        """
        Inject scaling function.

        Multiplies hidden states by scale factor.
        """
        return self.inject(
            layer_name=layer_name,
            injection_fn=lambda x: x * scale,
            name=f"{layer_name}_scale_{scale}"
        )

    def inject_additive(
        self,
        layer_name: str,
        bias: Union[torch.Tensor, float]
    ) -> str:
        """
        Inject additive bias.

        Adds a constant to hidden states.
        """
        return self.inject(
            layer_name=layer_name,
            injection_fn=lambda x: x + bias,
            name=f"{layer_name}_additive"
        )

    def inject_noise(
        self,
        layer_name: str,
        noise_scale: float = 0.01
    ) -> str:
        """
        Inject random noise (for exploration/creativity).
        """
        def add_noise(x):
            noise = torch.randn_like(x) * noise_scale
            return x + noise

        return self.inject(
            layer_name=layer_name,
            injection_fn=add_noise,
            name=f"{layer_name}_noise_{noise_scale}"
        )

    def inject_custom_layer(
        self,
        layer_name: str,
        custom_module: nn.Module
    ) -> str:
        """
        Inject a custom nn.Module.

        The module must accept and return tensors of same shape.
        """
        return self.inject(
            layer_name=layer_name,
            injection_fn=custom_module.forward,
            name=f"{layer_name}_custom"
        )

    def remove(self, name: str):
        """Remove a specific injection by name."""
        self.hook_manager.remove_hook(name)
        if name in self.active_injections:
            self.active_injections.remove(name)

    def remove_all(self):
        """Remove all active injections."""
        self.hook_manager.remove_all()
        self.active_injections.clear()

    def list_injections(self) -> List[str]:
        """List all active injection names."""
        return self.active_injections.copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()


# Convenience functions

def quick_inject(model, layer_idx: int, fn: Callable) -> LayerInjector:
    """
    Quick helper to inject at a specific layer index.

    Assumes GPT-2 style architecture (transformer.h.{idx}).

    Usage:
        with quick_inject(model, 6, lambda x: x * 0.9) as injector:
            output = model(input_ids)
    """
    injector = LayerInjector(model)
    injector.inject(f"transformer.h.{layer_idx}", fn)
    return injector
