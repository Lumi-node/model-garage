"""
Debate Chamber - Self-debate injection between layers.

Like a dual exhaust system - two paths that merge for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Callable, Optional


class DebateChamber(nn.Module):
    """
    A debate chamber that creates divergent perspectives and reconciles them.

    Can be injected between any two layers using LayerInjector.
    """

    def __init__(
        self,
        hidden_dim: int,
        divergence_method: str = "dropout",
        reconciliation_method: str = "average",
        divergence_strength: float = 0.1
    ):
        """
        Initialize debate chamber.

        Args:
            hidden_dim: Dimension of hidden states
            divergence_method: How to create different perspectives
                - "dropout": Different dropout masks
                - "perturbation": Add different noise
                - "projection": Different learned projections
            reconciliation_method: How to merge perspectives
                - "average": Simple mean
                - "confidence": Weight by magnitude
                - "gated": Learned gating
            divergence_strength: How different the perspectives should be
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.divergence_method = divergence_method
        self.reconciliation_method = reconciliation_method
        self.divergence_strength = divergence_strength

        # Setup divergence
        if divergence_method == "projection":
            self.proj_a = nn.Linear(hidden_dim, hidden_dim)
            self.proj_b = nn.Linear(hidden_dim, hidden_dim)
            # Initialize A as identity, B as perturbed
            nn.init.eye_(self.proj_a.weight)
            nn.init.zeros_(self.proj_a.bias)
            nn.init.eye_(self.proj_b.weight)
            with torch.no_grad():
                self.proj_b.weight.add_(torch.randn_like(self.proj_b.weight) * divergence_strength)
            nn.init.zeros_(self.proj_b.bias)

        # Setup reconciliation
        if reconciliation_method == "gated":
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def create_perspectives(
        self,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two different perspectives of the hidden states.
        """
        if self.divergence_method == "dropout":
            # Different dropout masks
            mask_a = torch.bernoulli(
                torch.ones_like(hidden) * (1 - self.divergence_strength)
            ) / (1 - self.divergence_strength)
            mask_b = torch.bernoulli(
                torch.ones_like(hidden) * (1 - self.divergence_strength)
            ) / (1 - self.divergence_strength)
            view_a = hidden * mask_a
            view_b = hidden * mask_b

        elif self.divergence_method == "perturbation":
            # Add different noise
            noise_a = torch.randn_like(hidden) * self.divergence_strength
            noise_b = torch.randn_like(hidden) * self.divergence_strength
            view_a = hidden + noise_a
            view_b = hidden + noise_b

        elif self.divergence_method == "projection":
            # Different learned projections
            view_a = self.proj_a(hidden)
            view_b = self.proj_b(hidden)

        else:
            raise ValueError(f"Unknown divergence method: {self.divergence_method}")

        return view_a, view_b

    def reconcile(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconcile two perspectives into one.
        """
        if self.reconciliation_method == "average":
            return (view_a + view_b) / 2

        elif self.reconciliation_method == "confidence":
            # Weight by magnitude (confidence)
            conf_a = view_a.abs().mean(dim=-1, keepdim=True)
            conf_b = view_b.abs().mean(dim=-1, keepdim=True)
            total = conf_a + conf_b + 1e-8
            return (conf_a / total) * view_a + (conf_b / total) * view_b

        elif self.reconciliation_method == "gated":
            # Learned gating
            combined = torch.cat([view_a, view_b], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            return gate * view_a + (1 - gate) * view_b

        else:
            raise ValueError(f"Unknown reconciliation method: {self.reconciliation_method}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply debate: create perspectives, reconcile, return result.
        """
        view_a, view_b = self.create_perspectives(hidden)
        return self.reconcile(view_a, view_b)

    def forward_with_info(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply debate and return additional info.
        """
        view_a, view_b = self.create_perspectives(hidden)
        reconciled = self.reconcile(view_a, view_b)

        # Compute metrics
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(
                view_a.flatten(), view_b.flatten(), dim=0
            ).item()
            l2_diff = (view_a - view_b).norm().item()

        info = {
            "cosine_similarity": cosine_sim,
            "l2_difference": l2_diff,
            "divergence_method": self.divergence_method,
            "reconciliation_method": self.reconciliation_method,
        }

        return reconciled, info


class SelfDebate:
    """
    High-level wrapper to add self-debate to any model.

    Usage:
        debate = SelfDebate(model, layer_idx=6)
        with debate:
            output = model(input_ids)  # Now uses debate at layer 6
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 6,
        divergence_method: str = "dropout",
        reconciliation_method: str = "average",
        divergence_strength: float = 0.1,
        layer_name_template: str = "transformer.h.{idx}"
    ):
        """
        Initialize self-debate wrapper.

        Args:
            model: The model to wrap
            layer_idx: Which layer to inject debate after
            divergence_method: How to create perspectives
            reconciliation_method: How to merge perspectives
            divergence_strength: How different perspectives should be
            layer_name_template: Template for layer names (GPT-2 style default)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.layer_name = layer_name_template.format(idx=layer_idx)

        # Get hidden dim from model
        config = model.config
        hidden_dim = getattr(config, "hidden_size", getattr(config, "n_embd", 768))

        # Create chamber
        device = next(model.parameters()).device
        self.chamber = DebateChamber(
            hidden_dim=hidden_dim,
            divergence_method=divergence_method,
            reconciliation_method=reconciliation_method,
            divergence_strength=divergence_strength
        ).to(device)

        # Hook handle
        self._hook_handle = None
        self._debate_info = []

    def _hook_fn(self, module, input, output):
        """Hook function that applies debate."""
        if isinstance(output, tuple):
            hidden = output[0]
            debated, info = self.chamber.forward_with_info(hidden)
            self._debate_info.append(info)
            return (debated,) + output[1:]
        else:
            debated, info = self.chamber.forward_with_info(output)
            self._debate_info.append(info)
            return debated

    def enable(self):
        """Enable debate."""
        if self._hook_handle is None:
            layer = self._get_layer(self.layer_name)
            self._hook_handle = layer.register_forward_hook(self._hook_fn)
        self._debate_info = []

    def disable(self):
        """Disable debate."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_debate_info(self):
        """Get info from last forward pass."""
        return self._debate_info.copy()

    def _get_layer(self, name: str):
        """Get layer by dot-separated name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
