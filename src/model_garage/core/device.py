"""
Device Utilities - Handle CUDA/CPU placement.

Like the garage's electrical system - makes sure power goes where needed.
"""

import torch
from typing import Union, Optional
from contextlib import contextmanager


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_device(
    tensor: torch.Tensor,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Move tensor to device if not already there."""
    target = torch.device(device) if isinstance(device, str) else device
    if tensor.device != target:
        return tensor.to(target)
    return tensor


class DeviceManager:
    """
    Manage device placement for a session.

    Provides consistent device handling across multiple operations.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

    def to(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to managed device."""
        return ensure_device(tensor, self.device)

    def to_dict(self, d: dict) -> dict:
        """Move all tensors in a dict to managed device."""
        return {
            k: self.to(v) if isinstance(v, torch.Tensor) else v
            for k, v in d.items()
        }

    @contextmanager
    def scope(self):
        """Context manager for device scope."""
        with torch.device(self.device):
            yield

    @property
    def is_gpu(self) -> bool:
        return self.device.type == "cuda"

    def memory_stats(self) -> dict:
        """Get GPU memory stats if available."""
        if not self.is_gpu:
            return {"device": "cpu"}
        return {
            "device": str(self.device),
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            "cached_mb": torch.cuda.memory_reserved(self.device) / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024 / 1024,
        }

    def clear_cache(self):
        """Clear GPU cache if on CUDA."""
        if self.is_gpu:
            torch.cuda.empty_cache()
