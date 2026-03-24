"""
Base model builder for assembling hybrid architectures.

Like the build sheet - plan your custom ride before turning a wrench.
"""

import logging
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class BaseModelBuilder(ABC):
    """
    Base class for model builders.

    Defines the interface for combining extracted components
    into new hybrid architectures.
    """

    def __init__(self, name: str, framework: str = "pytorch"):
        self.name = name
        self.framework = framework
        self.components = {}
        self.adapters = {}
        self.architecture = {}

    @abstractmethod
    def add_component(self, name: str, component: Any, metadata: Dict[str, Any]) -> None:
        """Add a component to the model."""
        pass

    @abstractmethod
    def add_adapter(self, source: str, target: str, adapter: Any) -> None:
        """Add an adapter between components to match dimensions."""
        pass

    @abstractmethod
    def define_forward_pass(self, execution_order: List[str]) -> None:
        """Define the execution order for the forward pass."""
        pass

    @abstractmethod
    def build(self) -> Any:
        """Build the complete model from components."""
        pass

    def validate_architecture(self) -> Tuple[bool, List[str]]:
        """Validate the model architecture."""
        errors = []

        if "execution_order" not in self.architecture:
            errors.append("No execution order defined")
            return False, errors

        for name in self.architecture["execution_order"]:
            if name not in self.components:
                errors.append(f"Component {name} in execution order doesn't exist")

        for (source, target) in self.adapters:
            if source not in self.components:
                errors.append(f"Source component {source} for adapter doesn't exist")
            if target not in self.components:
                errors.append(f"Target component {target} for adapter doesn't exist")

        return len(errors) == 0, errors

    def save_architecture(self, output_dir: str) -> str:
        """Save the model architecture specification."""
        serializable = {
            "name": self.name,
            "framework": self.framework,
            "components": {
                name: {
                    "type": meta.get("type", "unknown"),
                    "input_dim": meta.get("input_dim"),
                    "output_dim": meta.get("output_dim"),
                    "source": meta.get("source", "custom"),
                }
                for name, (_, meta) in self.components.items()
            },
            "execution_order": self.architecture.get("execution_order", []),
        }

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.name}_architecture.json")

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        return path
