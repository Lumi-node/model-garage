"""
Hook Manager - Register and manage PyTorch hooks.

Like the air hose connections - attach tools to the model safely.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class HookHandle:
    """Wrapper for a registered hook."""
    name: str
    layer_name: str
    hook_type: str  # "forward", "backward", "forward_pre"
    handle: Any  # torch hook handle

    def remove(self):
        """Remove this hook."""
        self.handle.remove()


class HookManager:
    """
    Centralized hook management for model manipulation.

    Features:
    - Named hooks for easy tracking
    - Automatic cleanup
    - Hook chaining
    - Debug logging
    """

    def __init__(self, model: nn.Module, debug: bool = False):
        self.model = model
        self.hooks: Dict[str, HookHandle] = {}
        self.debug = debug
        self._captured_data: Dict[str, Any] = {}

    def register_forward_hook(
        self,
        layer_name: str,
        hook_fn: Callable,
        hook_name: Optional[str] = None
    ) -> str:
        """
        Register a forward hook on a named layer.

        Args:
            layer_name: Name of layer (e.g., "transformer.h.6")
            hook_fn: Function(module, input, output) -> modified_output or None
            hook_name: Optional name for this hook

        Returns:
            Hook name for later reference
        """
        layer = self._get_layer(layer_name)
        hook_name = hook_name or f"{layer_name}_forward_{len(self.hooks)}"

        if self.debug:
            original_fn = hook_fn
            def hook_fn(module, input, output):
                print(f"[Hook] {hook_name} triggered on {layer_name}")
                return original_fn(module, input, output)

        handle = layer.register_forward_hook(hook_fn)

        self.hooks[hook_name] = HookHandle(
            name=hook_name,
            layer_name=layer_name,
            hook_type="forward",
            handle=handle
        )

        return hook_name

    def register_capture_hook(
        self,
        layer_name: str,
        hook_name: Optional[str] = None,
        capture_input: bool = False,
        capture_output: bool = True
    ) -> str:
        """
        Register a hook that captures activations without modifying them.

        Captured data accessible via get_captured(hook_name).
        """
        hook_name = hook_name or f"{layer_name}_capture_{len(self.hooks)}"

        def capture_fn(module, input, output):
            data = {}
            if capture_input:
                data["input"] = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
            if capture_output:
                data["output"] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            self._captured_data[hook_name] = data
            return None  # Don't modify

        return self.register_forward_hook(layer_name, capture_fn, hook_name)

    def register_injection_hook(
        self,
        layer_name: str,
        injection_fn: Callable[[torch.Tensor], torch.Tensor],
        hook_name: Optional[str] = None
    ) -> str:
        """
        Register a hook that modifies layer output.

        Args:
            layer_name: Name of layer to inject after
            injection_fn: Function(hidden_states) -> modified_hidden_states
            hook_name: Optional name
        """
        hook_name = hook_name or f"{layer_name}_inject_{len(self.hooks)}"

        def inject_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                modified = injection_fn(hidden)
                return (modified,) + output[1:]
            else:
                return injection_fn(output)

        return self.register_forward_hook(layer_name, inject_fn, hook_name)

    def get_captured(self, hook_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get data captured by a capture hook."""
        return self._captured_data.get(hook_name)

    def clear_captured(self):
        """Clear all captured data."""
        self._captured_data.clear()

    def remove_hook(self, hook_name: str):
        """Remove a specific hook by name."""
        if hook_name in self.hooks:
            self.hooks[hook_name].remove()
            del self.hooks[hook_name]

    def remove_all(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self._captured_data.clear()

    def list_hooks(self) -> List[str]:
        """List all registered hook names."""
        return list(self.hooks.keys())

    def _get_layer(self, layer_name: str) -> nn.Module:
        """Get a layer by dot-separated name."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup hooks on exit."""
        self.remove_all()
