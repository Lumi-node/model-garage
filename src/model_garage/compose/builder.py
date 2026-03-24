"""
PyTorch model builder for assembling hybrid architectures.

Like the engine builder's bench - mate parts from different donors.
"""

import logging
from typing import Dict, Any, List

from model_garage.compose.base import BaseModelBuilder

logger = logging.getLogger(__name__)


class PyTorchModelBuilder(BaseModelBuilder):
    """
    Builder for PyTorch hybrid models.

    Combines extracted components from different models
    into new architectures with dimension adapters.
    """

    def __init__(self, name: str):
        super().__init__(name, framework="pytorch")
        self.execution_graph = {}

    def add_component(self, name: str, component: Any, metadata: Dict[str, Any]) -> None:
        """
        Add a component to the model.

        Args:
            name: Name for this component in the architecture
            component: The component (PyTorch module or ExtractedComponent)
            metadata: Component metadata with input/output dimensions
        """
        if name in self.components:
            logger.warning(f"Component {name} already exists, overwriting")
        self.components[name] = (component, metadata)
        logger.info(
            f"Added {name}: {metadata.get('input_dim')}→{metadata.get('output_dim')}"
        )

    def add_adapter(self, source: str, target: str, adapter: Any) -> None:
        """
        Add a dimension adapter between components.

        Args:
            source: Name of source component
            target: Name of target component
            adapter: Linear adapter module
        """
        if source not in self.components:
            raise ValueError(f"Source component {source} does not exist")
        if target not in self.components:
            raise ValueError(f"Target component {target} does not exist")

        _, source_meta = self.components[source]
        _, target_meta = self.components[target]

        adapter_meta = {
            "input_dim": source_meta.get("output_dim"),
            "output_dim": target_meta.get("input_dim"),
        }

        self.adapters[(source, target)] = (adapter, adapter_meta)

        if source not in self.execution_graph:
            self.execution_graph[source] = []
        self.execution_graph[source].append(target)

    def define_forward_pass(self, execution_order: List[str]) -> None:
        """Define the execution order for the forward pass."""
        for name in execution_order:
            if name not in self.components:
                raise ValueError(f"Component {name} not found")
        self.architecture["execution_order"] = execution_order

    def build(self) -> Dict[str, Any]:
        """
        Build the model from components.

        Returns:
            Dict containing the assembled model specification.
        """
        is_valid, errors = self.validate_architecture()
        if not is_valid:
            raise ValueError(f"Invalid architecture: {'; '.join(errors)}")

        logger.info(f"Building {self.name}")

        return {
            "name": self.name,
            "components": {n: c[0] for n, c in self.components.items()},
            "adapters": {
                f"{s}->{t}": a[0] for (s, t), a in self.adapters.items()
            },
            "execution_order": self.architecture.get("execution_order", []),
        }
