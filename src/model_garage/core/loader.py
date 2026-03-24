"""
Model Loader - Consistent model loading across garages.

Like the car lift - gets the vehicle into position for work.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class ModelLoader:
    """
    Standardized model loading for Model Garage.

    Handles:
    - HuggingFace models
    - Device placement
    - Memory optimization
    - Model info extraction
    """

    SUPPORTED_ARCHITECTURES = {
        "gpt2": "GPT2LMHeadModel",
        "llama": "LlamaForCausalLM",
        "gemma": "GemmaForCausalLM",
        "phi": "PhiForCausalLM",
        "mistral": "MistralForCausalLM",
    }

    def __init__(self, device: Optional[str] = None):
        """
        Initialize loader.

        Args:
            device: Target device ("cuda", "cpu", "auto"). Default: auto-detect.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def load(
        self,
        model_id: str,
        load_tokenizer: bool = True,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
        """
        Load a model and optionally its tokenizer.

        Args:
            model_id: HuggingFace model ID or local path
            load_tokenizer: Whether to load tokenizer
            dtype: Optional dtype override (e.g., torch.float16)
            **kwargs: Additional args passed to from_pretrained

        Returns:
            (model, tokenizer, model_info)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine loading strategy based on device
        load_kwargs = {**kwargs}

        if self.device == "cuda":
            # Try to use GPU efficiently
            if dtype is None:
                dtype = torch.float16  # Default to fp16 on GPU

            load_kwargs["torch_dtype"] = dtype

            # Use device_map if model is large
            try:
                load_kwargs["device_map"] = "auto"
            except:
                pass  # Fall back to manual placement

        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except Exception as e:
            # Fallback: load without device_map
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model = model.to(self.device)

        model.eval()

        # Load tokenizer
        tokenizer = None
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # Extract model info
        model_info = self._extract_info(model, model_id)

        return model, tokenizer, model_info

    def _extract_info(self, model: Any, model_id: str) -> Dict[str, Any]:
        """Extract useful info about the model."""
        config = model.config

        info = {
            "model_id": model_id,
            "architecture": config.architectures[0] if hasattr(config, "architectures") and config.architectures else "unknown",
            "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
            "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", None)),
            "num_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", None)),
            "vocab_size": config.vocab_size,
            "max_position": getattr(config, "max_position_embeddings", getattr(config, "n_positions", None)),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        return info

    def get_layer_names(self, model: Any) -> Dict[str, str]:
        """
        Get standard layer names for a model.

        Returns dict mapping generic names to model-specific paths.
        """
        config = model.config
        arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else ""

        if "GPT2" in arch:
            n_layers = config.n_layer
            return {
                "embedding": "transformer.wte",
                "position_embedding": "transformer.wpe",
                "layers": [f"transformer.h.{i}" for i in range(n_layers)],
                "final_norm": "transformer.ln_f",
                "output_head": "lm_head",
            }
        elif "Llama" in arch or "Gemma" in arch or "Mistral" in arch:
            n_layers = config.num_hidden_layers
            return {
                "embedding": "model.embed_tokens",
                "layers": [f"model.layers.{i}" for i in range(n_layers)],
                "final_norm": "model.norm",
                "output_head": "lm_head",
            }
        elif "Phi" in arch:
            n_layers = config.num_hidden_layers
            return {
                "embedding": "model.embed_tokens",
                "layers": [f"model.layers.{i}" for i in range(n_layers)],
                "final_norm": "model.final_layernorm",
                "output_head": "lm_head",
            }
        else:
            # Generic fallback
            return {
                "note": f"Unknown architecture: {arch}. Inspect model manually.",
            }


def quick_load(model_id: str, device: Optional[str] = None):
    """
    Quick helper to load a model.

    Usage:
        model, tokenizer, info = quick_load("gpt2")
    """
    loader = ModelLoader(device)
    return loader.load(model_id)
