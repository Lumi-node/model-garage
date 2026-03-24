"""
Tensor Utilities - Shape manipulation and device handling.

Like the air compressor - provides pressure (compute) to all tools.
"""

import torch
from typing import Union, List, Tuple, Optional


class TensorUtils:
    """
    Common tensor operations used across all tools.
    """

    @staticmethod
    def ensure_device(tensor: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        """Move tensor to specified device if not already there."""
        if tensor.device != torch.device(device):
            return tensor.to(device)
        return tensor

    @staticmethod
    def ensure_shape(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Reshape tensor to target shape, handling common cases.

        Supports:
        - Adding batch dimension
        - Adding sequence dimension
        - Padding/truncating sequence length
        """
        current = tensor.shape
        target = target_shape

        # Add batch dim if missing
        if len(current) == len(target) - 1:
            tensor = tensor.unsqueeze(0)
            current = tensor.shape

        # Add sequence dim if missing
        if len(current) == len(target) - 1:
            tensor = tensor.unsqueeze(1)
            current = tensor.shape

        # Handle sequence length mismatch
        if len(current) == len(target) and current[1] != target[1]:
            if current[1] < target[1]:
                # Pad
                padding = torch.zeros(
                    current[0], target[1] - current[1], *current[2:],
                    device=tensor.device, dtype=tensor.dtype
                )
                tensor = torch.cat([tensor, padding], dim=1)
            else:
                # Truncate
                tensor = tensor[:, :target[1]]

        return tensor

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors."""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        return torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        ).item()

    @staticmethod
    def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute L2 distance between two tensors."""
        return (a - b).norm().item()

    @staticmethod
    def stats(tensor: torch.Tensor) -> dict:
        """Get basic statistics about a tensor."""
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": tensor.float().mean().item(),
            "std": tensor.float().std().item(),
            "min": tensor.float().min().item(),
            "max": tensor.float().max().item(),
            "sparsity": (tensor == 0).float().mean().item(),
        }

    @staticmethod
    def project(tensor: torch.Tensor, from_dim: int, to_dim: int) -> torch.Tensor:
        """
        Project tensor from one dimension to another using learned linear.

        Note: This creates a NEW projection each time. For reusable projections,
        use the Projector class instead.
        """
        projection = torch.nn.Linear(from_dim, to_dim, device=tensor.device)
        return projection(tensor)


class Projector:
    """
    Reusable dimension projector.

    Like an adapter socket - converts between different sizes.
    """

    def __init__(self, from_dim: int, to_dim: int, device: str = "cpu"):
        self.projection = torch.nn.Linear(from_dim, to_dim)
        self.projection = self.projection.to(device)
        self.from_dim = from_dim
        self.to_dim = to_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Project tensor to new dimension."""
        return self.projection(tensor)

    def save(self, path: str):
        """Save projector weights."""
        torch.save({
            "weights": self.projection.state_dict(),
            "from_dim": self.from_dim,
            "to_dim": self.to_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Projector":
        """Load projector from file."""
        data = torch.load(path, map_location=device)
        proj = cls(data["from_dim"], data["to_dim"], device)
        proj.projection.load_state_dict(data["weights"])
        return proj
