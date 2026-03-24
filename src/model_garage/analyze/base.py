"""
Base analyzer for studying neural pathways in models.

Like the diagnostic scanner - understand what each component contributes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Base class for model analyzers.

    Defines the interface for analyzing neural pathways to understand
    which components contribute most to model behavior.
    """

    def __init__(self, model: Any, framework: str = "pytorch"):
        self.model = model
        self.framework = framework
        self.activation_hooks = {}
        self.results = {}

    @abstractmethod
    def register_hooks(self) -> None:
        """Register hooks to capture activations during forward pass."""
        pass

    @abstractmethod
    def analyze_activations(self, inputs: Any, labels: Any) -> Dict[str, Any]:
        """Analyze activations in response to specific inputs."""
        pass

    @abstractmethod
    def identify_important_neurons(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Identify neurons that strongly correlate with target behavior."""
        pass

    def calculate_correlation(
        self,
        activations: List[Dict[str, Any]],
        outcomes: List[float],
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Calculate correlation between neuron activations and outcomes.

        Args:
            activations: List of activation maps per sample
            outcomes: List of outcome values

        Returns:
            Dict mapping layers to (neuron_idx, correlation) tuples
        """
        outcomes_array = np.array(outcomes)
        correlations = {}

        for layer_name in activations[0]:
            layer_acts = []
            for sample_idx in range(len(activations)):
                if layer_name in activations[sample_idx]:
                    layer_acts.append(activations[sample_idx][layer_name].flatten())

            if not layer_acts:
                continue

            layer_acts_array = np.array(layer_acts)
            layer_correlations = []

            for neuron_idx in range(layer_acts_array.shape[1]):
                neuron_acts = layer_acts_array[:, neuron_idx]
                try:
                    corr = np.corrcoef(neuron_acts, outcomes_array)[0, 1]
                    layer_correlations.append((neuron_idx, corr))
                except Exception as e:
                    logger.warning(
                        f"Correlation failed for neuron {neuron_idx} in {layer_name}: {e}"
                    )

            layer_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            correlations[layer_name] = layer_correlations

        return correlations

    def get_top_neurons(
        self,
        correlations: Dict[str, List[Tuple[int, float]]],
        top_n: int = 10,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Get the top neurons by correlation magnitude."""
        return {
            layer: sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
            for layer, corrs in correlations.items()
        }

    def save_results(self, output_path: str) -> str:
        """Save analysis results to disk."""
        import json
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        serializable = {}
        for key, value in self.results.items():
            serializable[key] = value if isinstance(value, dict) else str(value)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        return output_path
