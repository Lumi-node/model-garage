"""
Model Registry - Model-Agnostic Decomposition System
=====================================================

This is the foundation that makes everything fast.

PREPARE ONCE:
    Model → Decompose → Rust Bucket Parts → Registry

USE FOREVER:
    Pick parts → Run experiments → Mix models → Compare

Supported model families:
- GPT-2 (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Llama (llama-2-7b, llama-3-8b, etc.)
- Mistral (mistral-7b, mixtral-8x7b)
- Gemma (gemma-2b, gemma-7b, gemma-2-9b)
- Gemma 4 (gemma-4-E2B, E4B, 31B, 26B-A4B — multimodal MoE)
- Qwen (qwen-1.5-7b, qwen-2-7b)
- Phi (phi-2, phi-3-mini)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Type
from pathlib import Path
import json
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from enum import Enum


class ModelFamily(Enum):
    """Supported model architecture families."""
    GPT2 = "gpt2"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    GEMMA4 = "gemma4"
    QWEN = "qwen"
    PHI = "phi"
    UNKNOWN = "unknown"


class PartType(Enum):
    """Standard rust bucket part types."""
    EMBEDDING = "embedding"           # Token + position embeddings
    ATTENTION = "attention"           # Self-attention block
    FFN = "ffn"                       # Feed-forward network
    LAYER_NORM = "layer_norm"         # Normalization layer
    OUTPUT_HEAD = "output_head"       # Final projection to vocab
    ROTARY_EMB = "rotary_emb"         # RoPE embeddings (modern models)
    GATE = "gate"                     # MoE gating (Mixtral, etc.)
    FULL_LAYER = "full_layer"         # Complete transformer layer
    VISION_ENCODER = "vision_encoder" # Vision tower (SigLIP, etc.)
    AUDIO_ENCODER = "audio_encoder"   # Audio tower (USM, etc.)
    EXPERT = "expert"                 # Individual MoE expert


@dataclass
class PartSpec:
    """Specification for a model part (rust bucket component)."""
    part_type: PartType
    layer_idx: Optional[int]          # None for non-layer parts (embedding, output)
    module_path: str                  # Path in model (e.g., "transformer.h.0.attn")
    input_dim: int
    output_dim: int
    num_heads: Optional[int] = None   # For attention
    head_dim: Optional[int] = None    # For attention
    intermediate_dim: Optional[int] = None  # For FFN
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """Full specification of a decomposed model."""
    model_id: str                     # HuggingFace model ID
    family: ModelFamily
    hidden_dim: int
    num_layers: int
    num_heads: int
    vocab_size: int
    max_seq_len: int
    parts: Dict[str, PartSpec] = field(default_factory=dict)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def get_attention(self, layer_idx: int) -> Optional[PartSpec]:
        """Get attention part for a specific layer."""
        key = f"attention_{layer_idx}"
        return self.parts.get(key)

    def get_ffn(self, layer_idx: int) -> Optional[PartSpec]:
        """Get FFN part for a specific layer."""
        key = f"ffn_{layer_idx}"
        return self.parts.get(key)

    def get_full_layer(self, layer_idx: int) -> Optional[PartSpec]:
        """Get full layer part."""
        key = f"layer_{layer_idx}"
        return self.parts.get(key)

    def all_attention_parts(self) -> List[PartSpec]:
        """Get all attention parts."""
        return [p for p in self.parts.values() if p.part_type == PartType.ATTENTION]

    def all_ffn_parts(self) -> List[PartSpec]:
        """Get all FFN parts."""
        return [p for p in self.parts.values() if p.part_type == PartType.FFN]


class ModelDecomposer(ABC):
    """Abstract base for model-family-specific decomposers."""

    @abstractmethod
    def detect(self, model: nn.Module, model_id: str) -> bool:
        """Check if this decomposer handles the given model."""
        pass

    @abstractmethod
    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        """Decompose the model into parts."""
        pass

    @abstractmethod
    def get_module(self, model: nn.Module, part: PartSpec) -> nn.Module:
        """Get the actual module for a part."""
        pass


class GPT2Decomposer(ModelDecomposer):
    """Decomposer for GPT-2 family models."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return (
            "gpt2" in model_id.lower() or
            hasattr(model, "transformer") and hasattr(model.transformer, "h")
        )

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        config = model.config

        spec = ModelSpec(
            model_id=model_id,
            family=ModelFamily.GPT2,
            hidden_dim=config.n_embd,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            vocab_size=config.vocab_size,
            max_seq_len=config.n_positions,
        )

        # Embedding
        spec.parts["embedding"] = PartSpec(
            part_type=PartType.EMBEDDING,
            layer_idx=None,
            module_path="transformer.wte",
            input_dim=config.vocab_size,
            output_dim=config.n_embd,
        )

        # Position embedding
        spec.parts["pos_embedding"] = PartSpec(
            part_type=PartType.EMBEDDING,
            layer_idx=None,
            module_path="transformer.wpe",
            input_dim=config.n_positions,
            output_dim=config.n_embd,
        )

        # Layers
        for i in range(config.n_layer):
            # Attention
            spec.parts[f"attention_{i}"] = PartSpec(
                part_type=PartType.ATTENTION,
                layer_idx=i,
                module_path=f"transformer.h.{i}.attn",
                input_dim=config.n_embd,
                output_dim=config.n_embd,
                num_heads=config.n_head,
                head_dim=config.n_embd // config.n_head,
            )

            # FFN
            spec.parts[f"ffn_{i}"] = PartSpec(
                part_type=PartType.FFN,
                layer_idx=i,
                module_path=f"transformer.h.{i}.mlp",
                input_dim=config.n_embd,
                output_dim=config.n_embd,
                intermediate_dim=config.n_embd * 4,
            )

            # Layer norms
            spec.parts[f"ln_1_{i}"] = PartSpec(
                part_type=PartType.LAYER_NORM,
                layer_idx=i,
                module_path=f"transformer.h.{i}.ln_1",
                input_dim=config.n_embd,
                output_dim=config.n_embd,
            )

            spec.parts[f"ln_2_{i}"] = PartSpec(
                part_type=PartType.LAYER_NORM,
                layer_idx=i,
                module_path=f"transformer.h.{i}.ln_2",
                input_dim=config.n_embd,
                output_dim=config.n_embd,
            )

            # Full layer
            spec.parts[f"layer_{i}"] = PartSpec(
                part_type=PartType.FULL_LAYER,
                layer_idx=i,
                module_path=f"transformer.h.{i}",
                input_dim=config.n_embd,
                output_dim=config.n_embd,
                num_heads=config.n_head,
                intermediate_dim=config.n_embd * 4,
            )

        # Final layer norm
        spec.parts["ln_f"] = PartSpec(
            part_type=PartType.LAYER_NORM,
            layer_idx=None,
            module_path="transformer.ln_f",
            input_dim=config.n_embd,
            output_dim=config.n_embd,
        )

        # Output head
        spec.parts["output_head"] = PartSpec(
            part_type=PartType.OUTPUT_HEAD,
            layer_idx=None,
            module_path="lm_head",
            input_dim=config.n_embd,
            output_dim=config.vocab_size,
        )

        return spec

    def get_module(self, model: nn.Module, part: PartSpec) -> nn.Module:
        """Navigate to the module using the path."""
        module = model
        for attr in part.module_path.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module


class LlamaDecomposer(ModelDecomposer):
    """Decomposer for Llama/Llama-2/Llama-3 family models."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return (
            "llama" in model_id.lower() or
            hasattr(model, "model") and hasattr(model.model, "layers")
        )

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        config = model.config

        spec = ModelSpec(
            model_id=model_id,
            family=ModelFamily.LLAMA,
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
            max_seq_len=getattr(config, "max_position_embeddings", 4096),
            extra_info={
                "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "rope_theta": getattr(config, "rope_theta", 10000.0),
            }
        )

        # Embedding
        spec.parts["embedding"] = PartSpec(
            part_type=PartType.EMBEDDING,
            layer_idx=None,
            module_path="model.embed_tokens",
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
        )

        # Layers
        for i in range(config.num_hidden_layers):
            head_dim = config.hidden_size // config.num_attention_heads

            # Attention
            spec.parts[f"attention_{i}"] = PartSpec(
                part_type=PartType.ATTENTION,
                layer_idx=i,
                module_path=f"model.layers.{i}.self_attn",
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                extra_info={
                    "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                }
            )

            # FFN (SwiGLU in Llama)
            intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
            spec.parts[f"ffn_{i}"] = PartSpec(
                part_type=PartType.FFN,
                layer_idx=i,
                module_path=f"model.layers.{i}.mlp",
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                intermediate_dim=intermediate_size,
                extra_info={"activation": "silu"}
            )

            # RMSNorm layers
            spec.parts[f"input_layernorm_{i}"] = PartSpec(
                part_type=PartType.LAYER_NORM,
                layer_idx=i,
                module_path=f"model.layers.{i}.input_layernorm",
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                extra_info={"type": "rmsnorm"}
            )

            spec.parts[f"post_attention_layernorm_{i}"] = PartSpec(
                part_type=PartType.LAYER_NORM,
                layer_idx=i,
                module_path=f"model.layers.{i}.post_attention_layernorm",
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                extra_info={"type": "rmsnorm"}
            )

            # Full layer
            spec.parts[f"layer_{i}"] = PartSpec(
                part_type=PartType.FULL_LAYER,
                layer_idx=i,
                module_path=f"model.layers.{i}",
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_dim=intermediate_size,
            )

        # Final norm
        spec.parts["norm"] = PartSpec(
            part_type=PartType.LAYER_NORM,
            layer_idx=None,
            module_path="model.norm",
            input_dim=config.hidden_size,
            output_dim=config.hidden_size,
            extra_info={"type": "rmsnorm"}
        )

        # Output head (lm_head)
        spec.parts["output_head"] = PartSpec(
            part_type=PartType.OUTPUT_HEAD,
            layer_idx=None,
            module_path="lm_head",
            input_dim=config.hidden_size,
            output_dim=config.vocab_size,
        )

        return spec

    def get_module(self, model: nn.Module, part: PartSpec) -> nn.Module:
        module = model
        for attr in part.module_path.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module


class MistralDecomposer(LlamaDecomposer):
    """Decomposer for Mistral family - same structure as Llama."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return "mistral" in model_id.lower()

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        spec = super().decompose(model, model_id)
        spec.family = ModelFamily.MISTRAL
        spec.extra_info["sliding_window"] = getattr(model.config, "sliding_window", None)
        return spec


class GemmaDecomposer(LlamaDecomposer):
    """Decomposer for Gemma family - similar to Llama with some differences."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return "gemma" in model_id.lower()

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        spec = super().decompose(model, model_id)
        spec.family = ModelFamily.GEMMA
        return spec


class Gemma4Decomposer(ModelDecomposer):
    """
    Decomposer for Gemma 4 family — multimodal MoE models.

    Gemma 4 variants:
    - E2B (5.1B total, 2.3B effective): 35 layers, text+image+audio
    - E4B (8B total, 4.5B effective): 42 layers, text+image+audio
    - 31B Dense (30.7B): 60 layers, text+image only
    - 26B-A4B (27B MoE): MoE variant with selective activation

    Architecture features:
    - Sliding window attention (512 or 1024 tokens)
    - 128K-256K context length
    - SigLIP vision encoder (~150M or ~550M params)
    - USM audio encoder (~300M params, not on 31B)
    - 262K vocabulary (largest in Gemma family)
    """

    def detect(self, model: nn.Module, model_id: str) -> bool:
        model_lower = model_id.lower()
        return "gemma-4" in model_lower or "gemma4" in model_lower

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        config = model.config

        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        vocab_size = config.vocab_size

        # Detect MoE vs Dense
        is_moe = hasattr(config, "num_local_experts") or "A4B" in model_id
        num_experts = getattr(config, "num_local_experts", None)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", None)

        # Detect modalities
        has_vision = hasattr(model, "vision_tower") or hasattr(model, "vision_encoder")
        has_audio = hasattr(model, "audio_tower") or hasattr(model, "audio_encoder")

        spec = ModelSpec(
            model_id=model_id,
            family=ModelFamily.GEMMA4,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            max_seq_len=getattr(config, "max_position_embeddings", 131072),
            extra_info={
                "num_kv_heads": getattr(config, "num_key_value_heads", num_heads),
                "sliding_window": getattr(config, "sliding_window", None),
                "is_moe": is_moe,
                "num_experts": num_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "has_vision": has_vision,
                "has_audio": has_audio,
                "rope_theta": getattr(config, "rope_theta", 10000.0),
            }
        )

        # Embedding
        spec.parts["embedding"] = PartSpec(
            part_type=PartType.EMBEDDING,
            layer_idx=None,
            module_path="model.embed_tokens",
            input_dim=vocab_size,
            output_dim=hidden_size,
        )

        # Vision encoder (if present)
        if has_vision:
            vision_path = "vision_tower" if hasattr(model, "vision_tower") else "vision_encoder"
            vision_params = sum(
                p.numel() for p in getattr(model, vision_path).parameters()
            ) if hasattr(model, vision_path) else 0
            spec.parts["vision_encoder"] = PartSpec(
                part_type=PartType.VISION_ENCODER,
                layer_idx=None,
                module_path=vision_path,
                input_dim=0,
                output_dim=hidden_size,
                extra_info={"total_params": vision_params}
            )

        # Audio encoder (if present)
        if has_audio:
            audio_path = "audio_tower" if hasattr(model, "audio_tower") else "audio_encoder"
            audio_params = sum(
                p.numel() for p in getattr(model, audio_path).parameters()
            ) if hasattr(model, audio_path) else 0
            spec.parts["audio_encoder"] = PartSpec(
                part_type=PartType.AUDIO_ENCODER,
                layer_idx=None,
                module_path=audio_path,
                input_dim=0,
                output_dim=hidden_size,
                extra_info={"total_params": audio_params}
            )

        # Transformer layers
        for i in range(num_layers):
            head_dim = hidden_size // num_heads

            # Attention
            spec.parts[f"attention_{i}"] = PartSpec(
                part_type=PartType.ATTENTION,
                layer_idx=i,
                module_path=f"model.layers.{i}.self_attn",
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                extra_info={
                    "num_kv_heads": getattr(
                        config, "num_key_value_heads", num_heads
                    ),
                }
            )

            # FFN — MoE or Dense
            if is_moe and num_experts:
                # MoE gate
                spec.parts[f"gate_{i}"] = PartSpec(
                    part_type=PartType.GATE,
                    layer_idx=i,
                    module_path=f"model.layers.{i}.block_sparse_moe.gate",
                    input_dim=hidden_size,
                    output_dim=num_experts,
                    extra_info={
                        "num_experts": num_experts,
                        "top_k": num_experts_per_tok,
                    }
                )
                # Individual experts
                intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
                for e in range(num_experts):
                    spec.parts[f"expert_{i}_{e}"] = PartSpec(
                        part_type=PartType.EXPERT,
                        layer_idx=i,
                        module_path=f"model.layers.{i}.block_sparse_moe.experts.{e}",
                        input_dim=hidden_size,
                        output_dim=hidden_size,
                        intermediate_dim=intermediate_size,
                        extra_info={"expert_idx": e}
                    )
            else:
                # Dense FFN
                intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
                spec.parts[f"ffn_{i}"] = PartSpec(
                    part_type=PartType.FFN,
                    layer_idx=i,
                    module_path=f"model.layers.{i}.mlp",
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    intermediate_dim=intermediate_size,
                    extra_info={"activation": "gelu"}
                )

            # RMSNorm layers
            spec.parts[f"input_layernorm_{i}"] = PartSpec(
                part_type=PartType.LAYER_NORM,
                layer_idx=i,
                module_path=f"model.layers.{i}.input_layernorm",
                input_dim=hidden_size,
                output_dim=hidden_size,
                extra_info={"type": "rmsnorm"}
            )

            spec.parts[f"post_attention_layernorm_{i}"] = PartSpec(
                part_type=PartType.LAYER_NORM,
                layer_idx=i,
                module_path=f"model.layers.{i}.post_attention_layernorm",
                input_dim=hidden_size,
                output_dim=hidden_size,
                extra_info={"type": "rmsnorm"}
            )

            # Full layer
            spec.parts[f"layer_{i}"] = PartSpec(
                part_type=PartType.FULL_LAYER,
                layer_idx=i,
                module_path=f"model.layers.{i}",
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                intermediate_dim=getattr(config, "intermediate_size", hidden_size * 4),
            )

        # Final norm
        spec.parts["norm"] = PartSpec(
            part_type=PartType.LAYER_NORM,
            layer_idx=None,
            module_path="model.norm",
            input_dim=hidden_size,
            output_dim=hidden_size,
            extra_info={"type": "rmsnorm"}
        )

        # Output head
        spec.parts["output_head"] = PartSpec(
            part_type=PartType.OUTPUT_HEAD,
            layer_idx=None,
            module_path="lm_head",
            input_dim=hidden_size,
            output_dim=vocab_size,
        )

        return spec

    def get_module(self, model: nn.Module, part: PartSpec) -> nn.Module:
        module = model
        for attr in part.module_path.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module


class QwenDecomposer(LlamaDecomposer):
    """Decomposer for Qwen family - similar to Llama."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return "qwen" in model_id.lower()

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        spec = super().decompose(model, model_id)
        spec.family = ModelFamily.QWEN
        return spec


class PhiDecomposer(ModelDecomposer):
    """Decomposer for Phi family models."""

    def detect(self, model: nn.Module, model_id: str) -> bool:
        return "phi" in model_id.lower()

    def decompose(self, model: nn.Module, model_id: str) -> ModelSpec:
        config = model.config

        # Phi-2 vs Phi-3 have slightly different configs
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 2560))
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 32))
        num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 32))
        vocab_size = config.vocab_size

        spec = ModelSpec(
            model_id=model_id,
            family=ModelFamily.PHI,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            max_seq_len=getattr(config, "max_position_embeddings", 2048),
        )

        # Determine the model structure (phi-2 vs phi-3)
        if hasattr(model, "model"):
            # Phi-3 style
            base_path = "model"
            layers_attr = "layers"
        else:
            # Phi-2 style
            base_path = "transformer"
            layers_attr = "h"

        # Embedding
        embed_path = f"{base_path}.embed_tokens" if hasattr(model, "model") else f"{base_path}.embd.wte"
        spec.parts["embedding"] = PartSpec(
            part_type=PartType.EMBEDDING,
            layer_idx=None,
            module_path=embed_path,
            input_dim=vocab_size,
            output_dim=hidden_size,
        )

        # Layers
        for i in range(num_layers):
            layer_path = f"{base_path}.{layers_attr}.{i}"

            # Attention
            attn_path = f"{layer_path}.self_attn" if "model" in base_path else f"{layer_path}.mixer"
            spec.parts[f"attention_{i}"] = PartSpec(
                part_type=PartType.ATTENTION,
                layer_idx=i,
                module_path=attn_path,
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                head_dim=hidden_size // num_heads,
            )

            # FFN
            ffn_path = f"{layer_path}.mlp"
            intermediate = getattr(config, "intermediate_size", hidden_size * 4)
            spec.parts[f"ffn_{i}"] = PartSpec(
                part_type=PartType.FFN,
                layer_idx=i,
                module_path=ffn_path,
                input_dim=hidden_size,
                output_dim=hidden_size,
                intermediate_dim=intermediate,
            )

            # Full layer
            spec.parts[f"layer_{i}"] = PartSpec(
                part_type=PartType.FULL_LAYER,
                layer_idx=i,
                module_path=layer_path,
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                intermediate_dim=intermediate,
            )

        # Output head
        spec.parts["output_head"] = PartSpec(
            part_type=PartType.OUTPUT_HEAD,
            layer_idx=None,
            module_path="lm_head",
            input_dim=hidden_size,
            output_dim=vocab_size,
        )

        return spec

    def get_module(self, model: nn.Module, part: PartSpec) -> nn.Module:
        module = model
        for attr in part.module_path.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module


class ModelRegistry:
    """
    Central registry for decomposed models.

    Usage:
        registry = ModelRegistry()

        # Decompose and register a model
        spec = registry.register("meta-llama/Llama-2-7b-hf", model)

        # Get parts for experiments
        attn_layer_5 = registry.get_part("meta-llama/Llama-2-7b-hf", "attention_5")

        # List all registered models
        models = registry.list_models()

        # Compare architectures
        registry.compare("gpt2", "meta-llama/Llama-2-7b-hf")
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.decomposers: List[ModelDecomposer] = [
            GPT2Decomposer(),
            LlamaDecomposer(),
            MistralDecomposer(),
            Gemma4Decomposer(),  # Must be before GemmaDecomposer (gemma-4 matches gemma)
            GemmaDecomposer(),
            QwenDecomposer(),
            PhiDecomposer(),
        ]
        self.specs: Dict[str, ModelSpec] = {}
        self.models: Dict[str, nn.Module] = {}  # Cached loaded models
        self.cache_dir = cache_dir or Path.home() / ".model_garage" / "registry"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def detect_family(self, model: nn.Module, model_id: str) -> ModelFamily:
        """Detect which family a model belongs to."""
        for decomposer in self.decomposers:
            if decomposer.detect(model, model_id):
                return decomposer.decompose(model, model_id).family
        return ModelFamily.UNKNOWN

    def register(self, model_id: str, model: nn.Module) -> ModelSpec:
        """Decompose and register a model."""
        for decomposer in self.decomposers:
            if decomposer.detect(model, model_id):
                spec = decomposer.decompose(model, model_id)
                self.specs[model_id] = spec
                self.models[model_id] = model
                self._save_spec(spec)
                return spec

        raise ValueError(f"No decomposer found for model: {model_id}")

    def get_spec(self, model_id: str) -> Optional[ModelSpec]:
        """Get the spec for a registered model."""
        if model_id in self.specs:
            return self.specs[model_id]
        # Try loading from cache
        return self._load_spec(model_id)

    def get_part(self, model_id: str, part_name: str) -> Optional[PartSpec]:
        """Get a specific part from a registered model."""
        spec = self.get_spec(model_id)
        if spec:
            return spec.parts.get(part_name)
        return None

    def get_module(self, model_id: str, part_name: str) -> Optional[nn.Module]:
        """Get the actual module for a part."""
        if model_id not in self.models:
            return None
        spec = self.get_spec(model_id)
        if not spec or part_name not in spec.parts:
            return None

        part = spec.parts[part_name]
        for decomposer in self.decomposers:
            if decomposer.detect(self.models[model_id], model_id):
                return decomposer.get_module(self.models[model_id], part)
        return None

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.specs.keys())

    def list_parts(self, model_id: str, part_type: Optional[PartType] = None) -> List[str]:
        """List all parts for a model, optionally filtered by type."""
        spec = self.get_spec(model_id)
        if not spec:
            return []
        if part_type:
            return [name for name, part in spec.parts.items() if part.part_type == part_type]
        return list(spec.parts.keys())

    def compare(self, model_a: str, model_b: str) -> Dict[str, Any]:
        """Compare two model architectures."""
        spec_a = self.get_spec(model_a)
        spec_b = self.get_spec(model_b)

        if not spec_a or not spec_b:
            return {"error": "One or both models not registered"}

        return {
            "models": [model_a, model_b],
            "families": [spec_a.family.value, spec_b.family.value],
            "hidden_dims": [spec_a.hidden_dim, spec_b.hidden_dim],
            "num_layers": [spec_a.num_layers, spec_b.num_layers],
            "num_heads": [spec_a.num_heads, spec_b.num_heads],
            "vocab_sizes": [spec_a.vocab_size, spec_b.vocab_size],
            "compatible_parts": self._find_compatible_parts(spec_a, spec_b),
        }

    def _find_compatible_parts(self, spec_a: ModelSpec, spec_b: ModelSpec) -> Dict[str, List[str]]:
        """Find parts that could potentially be swapped between models."""
        compatible = {
            "same_dim": [],
            "attention_compatible": [],
            "ffn_compatible": [],
        }

        # Same hidden dimension = most compatible
        if spec_a.hidden_dim == spec_b.hidden_dim:
            compatible["same_dim"] = ["All parts potentially swappable"]

        # Check attention compatibility
        if spec_a.num_heads == spec_b.num_heads:
            compatible["attention_compatible"] = [
                f"attention_0 through attention_{min(spec_a.num_layers, spec_b.num_layers) - 1}"
            ]

        return compatible

    def _save_spec(self, spec: ModelSpec) -> None:
        """Save spec to cache."""
        safe_id = spec.model_id.replace("/", "__")
        path = self.cache_dir / f"{safe_id}.json"

        data = {
            "model_id": spec.model_id,
            "family": spec.family.value,
            "hidden_dim": spec.hidden_dim,
            "num_layers": spec.num_layers,
            "num_heads": spec.num_heads,
            "vocab_size": spec.vocab_size,
            "max_seq_len": spec.max_seq_len,
            "extra_info": spec.extra_info,
            "parts": {
                name: {
                    "part_type": part.part_type.value,
                    "layer_idx": part.layer_idx,
                    "module_path": part.module_path,
                    "input_dim": part.input_dim,
                    "output_dim": part.output_dim,
                    "num_heads": part.num_heads,
                    "head_dim": part.head_dim,
                    "intermediate_dim": part.intermediate_dim,
                    "extra_info": part.extra_info,
                }
                for name, part in spec.parts.items()
            }
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_spec(self, model_id: str) -> Optional[ModelSpec]:
        """Load spec from cache."""
        safe_id = model_id.replace("/", "__")
        path = self.cache_dir / f"{safe_id}.json"

        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        spec = ModelSpec(
            model_id=data["model_id"],
            family=ModelFamily(data["family"]),
            hidden_dim=data["hidden_dim"],
            num_layers=data["num_layers"],
            num_heads=data["num_heads"],
            vocab_size=data["vocab_size"],
            max_seq_len=data["max_seq_len"],
            extra_info=data.get("extra_info", {}),
        )

        for name, part_data in data["parts"].items():
            spec.parts[name] = PartSpec(
                part_type=PartType(part_data["part_type"]),
                layer_idx=part_data["layer_idx"],
                module_path=part_data["module_path"],
                input_dim=part_data["input_dim"],
                output_dim=part_data["output_dim"],
                num_heads=part_data.get("num_heads"),
                head_dim=part_data.get("head_dim"),
                intermediate_dim=part_data.get("intermediate_dim"),
                extra_info=part_data.get("extra_info", {}),
            )

        self.specs[model_id] = spec
        return spec


# Convenience function for quick registration
def register_model(model_id: str, model: nn.Module, registry: Optional[ModelRegistry] = None) -> ModelSpec:
    """Quick registration of a model."""
    if registry is None:
        registry = ModelRegistry()
    return registry.register(model_id, model)


# Example usage
if __name__ == "__main__":
    print("Model Registry - Model-Agnostic Decomposition System")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from model_garage.registry.models import ModelRegistry")
    print()
    print("  registry = ModelRegistry()")
    print("  spec = registry.register('gpt2', model)")
    print()
    print("  # Get any part")
    print("  attn = registry.get_part('gpt2', 'attention_5')")
    print()
    print("  # Compare models")
    print("  diff = registry.compare('gpt2', 'meta-llama/Llama-2-7b-hf')")
    print()
    print("Supported families:")
    for family in ModelFamily:
        if family != ModelFamily.UNKNOWN:
            print(f"  - {family.value}")
