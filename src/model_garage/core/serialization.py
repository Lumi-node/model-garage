"""
Serialization Utilities - Save and load components.

Like parts bins with labels - know what you have and where it came from.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ComponentMetadata:
    """Metadata for a saved component. Tracks provenance and compatibility."""
    component_type: str
    source_model: str
    layer_index: Optional[int]
    extraction_date: str
    toolkit_version: str
    hidden_dim: Optional[int]
    num_heads: Optional[int]
    compatible_with: list
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentMetadata":
        return cls(**d)


def save_component(
    component: Union[torch.nn.Module, torch.Tensor, Dict[str, torch.Tensor]],
    path: Union[str, Path],
    metadata: Optional[ComponentMetadata] = None,
    **extra_metadata,
) -> Path:
    """
    Save a component with metadata.

    Args:
        component: Module, Tensor, or state dict to save
        path: Directory to save to
        metadata: Optional ComponentMetadata
        **extra_metadata: Additional metadata fields

    Returns:
        Path to saved component directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(component, torch.nn.Module):
        state_dict = component.state_dict()
        component_type = "module"
    elif isinstance(component, torch.Tensor):
        state_dict = {"tensor": component}
        component_type = "tensor"
    elif isinstance(component, dict):
        state_dict = component
        component_type = "state_dict"
    else:
        raise ValueError(f"Cannot save component of type {type(component)}")

    torch.save(state_dict, path / "weights.pt")

    config = {
        "component_type": component_type,
        "saved_at": datetime.now().isoformat(),
        "toolkit_version": "0.1.0",
        "shapes": {k: list(v.shape) for k, v in state_dict.items()},
        "dtypes": {k: str(v.dtype) for k, v in state_dict.items()},
    }

    if metadata:
        config["metadata"] = metadata.to_dict()
    config.update(extra_metadata)

    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return path


def load_component(
    path: Union[str, Path],
    device: str = "cpu",
    return_metadata: bool = False,
) -> Union[Dict[str, torch.Tensor], tuple]:
    """
    Load a saved component.

    Args:
        path: Directory containing saved component
        device: Device to load to
        return_metadata: If True, also return metadata

    Returns:
        state_dict, or (state_dict, metadata) if return_metadata=True
    """
    path = Path(path)
    state_dict = torch.load(path / "weights.pt", map_location=device, weights_only=True)

    if not return_metadata:
        return state_dict

    config_path = path / "config.json"
    metadata = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        raw = config.get("metadata")
        if raw:
            metadata = ComponentMetadata.from_dict(raw)

    return state_dict, metadata
