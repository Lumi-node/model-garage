"""
Base component extractor for model garage.

This module defines the base class for extracting components from pre-trained models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """
    Base class for model component extractors.

    This class defines the interface for extracting components from
    pre-trained models. Specific model types should subclass this.
    """

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            model_name: Name or path of the pre-trained model
            cache_dir: Directory to cache downloaded models (optional)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.available_components = {}

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the pre-trained model.

        Returns:
            The loaded model
        """
        pass

    @abstractmethod
    def list_available_components(self) -> Dict[str, Any]:
        """
        List all available components that can be extracted.

        Returns:
            Dict mapping component names to their specifications
        """
        pass

    @abstractmethod
    def extract_component(self, component_name: str) -> Any:
        """
        Extract a specific component from the model.

        Args:
            component_name: Name of the component to extract

        Returns:
            The extracted component
        """
        pass

    def get_component_metadata(self, component_name: str) -> Dict[str, Any]:
        """
        Get metadata about a specific component.

        Args:
            component_name: Name of the component

        Returns:
            Dict containing metadata about the component
        """
        components = self.list_available_components()
        if component_name not in components:
            raise ValueError(f"Component {component_name} not found")

        return components[component_name]

    def __str__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(model_name={self.model_name})"

    def __repr__(self) -> str:
        """Detailed string representation of the extractor."""
        return f"{self.__class__.__name__}(model_name={self.model_name}, cache_dir={self.cache_dir})"
