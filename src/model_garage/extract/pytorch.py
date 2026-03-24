"""
Real PyTorch Transformer Component Extractor for Model Garage.

Extracts actual nn.Module components from HuggingFace transformers models.
Supports: Llama, Mistral, GPT-2, Gemma, BERT, and other standard architectures.

Usage:
    extractor = PyTorchExtractor("gpt2")
    extractor.load_model()

    # List what's available
    print(extractor.list_available_components())

    # Extract attention from layer 4
    attn, info = extractor.extract_component("self_attention", layer_idx=4)

    # attn is a real nn.Module you can use
    output = attn(hidden_states)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    """Specification for an extractable component."""
    name: str
    component_type: str  # attention, ffn, embedding, norm, etc.
    module_path: str     # Path to module in model (e.g., "model.layers.0.self_attn")
    input_dim: int
    output_dim: int
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_dim: Optional[int] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedComponent:
    """A component extracted from a model."""
    module: nn.Module
    spec: ComponentSpec
    source_model: str
    source_layer: int
    state_dict: Dict[str, torch.Tensor]

    def to(self, device: Union[str, torch.device]) -> 'ExtractedComponent':
        """Move component to device."""
        self.module = self.module.to(device)
        return self

    def parameters(self):
        """Get component parameters."""
        return self.module.parameters()

    def forward(self, *args, **kwargs):
        """Forward pass through component."""
        return self.module(*args, **kwargs)


class PyTorchExtractor:
    """
    Real PyTorch extractor for transformer models.

    Extracts actual nn.Module components that can be:
    - Run in isolation
    - Transplanted to other models
    - Analyzed and modified
    - Combined with other components
    """

    # Architecture patterns for different model families
    ARCH_PATTERNS = {
        'llama': {
            'layers_path': 'model.layers',
            'attention': 'self_attn',
            'ffn': 'mlp',
            'input_norm': 'input_layernorm',
            'post_attn_norm': 'post_attention_layernorm',
            'embed': 'model.embed_tokens',
            'lm_head': 'lm_head',
        },
        'mistral': {
            'layers_path': 'model.layers',
            'attention': 'self_attn',
            'ffn': 'mlp',
            'input_norm': 'input_layernorm',
            'post_attn_norm': 'post_attention_layernorm',
            'embed': 'model.embed_tokens',
            'lm_head': 'lm_head',
        },
        'gpt2': {
            'layers_path': 'transformer.h',
            'attention': 'attn',
            'ffn': 'mlp',
            'input_norm': 'ln_1',
            'post_attn_norm': 'ln_2',
            'embed': 'transformer.wte',
            'pos_embed': 'transformer.wpe',
            'lm_head': 'lm_head',
        },
        'gemma': {
            'layers_path': 'model.layers',
            'attention': 'self_attn',
            'ffn': 'mlp',
            'input_norm': 'input_layernorm',
            'post_attn_norm': 'post_attention_layernorm',
            'embed': 'model.embed_tokens',
            'lm_head': 'lm_head',
        },
        'bert': {
            'layers_path': 'encoder.layer',
            'attention': 'attention',
            'ffn': 'intermediate',  # BERT has intermediate + output
            'input_norm': 'attention.output.LayerNorm',
            'post_attn_norm': 'output.LayerNorm',
            'embed': 'embeddings',
        },
    }

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ):
        """
        Initialize the PyTorch extractor.

        Args:
            model_name: HuggingFace model name or local path
            cache_dir: Directory to cache downloaded models
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights (None = auto)
            trust_remote_code: Whether to trust remote code for custom models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.config = None
        self.tokenizer = None
        self.arch_type = None
        self.arch_pattern = None
        self._components_cache: Dict[str, ExtractedComponent] = {}

    def load_model(self, load_tokenizer: bool = True) -> nn.Module:
        """
        Load the pre-trained model.

        Args:
            load_tokenizer: Whether to also load the tokenizer

        Returns:
            The loaded model
        """
        try:
            from transformers import AutoModelForCausalLM, AutoModel, AutoConfig, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")

        logger.info(f"Loading model: {self.model_name}")

        # Load config first to determine architecture
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code
        )

        # Determine architecture type
        self.arch_type = self._detect_architecture()
        self.arch_pattern = self.ARCH_PATTERNS.get(self.arch_type, self.ARCH_PATTERNS['llama'])

        logger.info(f"Detected architecture: {self.arch_type}")

        # Load model
        load_kwargs = {
            'cache_dir': self.cache_dir,
            'trust_remote_code': self.trust_remote_code,
        }

        if self.torch_dtype:
            load_kwargs['torch_dtype'] = self.torch_dtype

        if self.device == "auto":
            load_kwargs['device_map'] = 'auto'

        # Try CausalLM first, fall back to base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs
            )

        if self.device not in ("auto", None) and self.device != "cpu":
            self.model = self.model.to(self.device)

        # Load tokenizer
        if load_tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=self.trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")

        logger.info(f"Model loaded successfully. {self._get_model_info()}")
        return self.model

    def _detect_architecture(self) -> str:
        """Detect the model architecture type."""
        model_type = getattr(self.config, 'model_type', '').lower()

        # Direct matches
        if 'llama' in model_type:
            return 'llama'
        elif 'mistral' in model_type:
            return 'mistral'
        elif 'gpt2' in model_type or 'gpt-2' in model_type:
            return 'gpt2'
        elif 'gemma' in model_type:
            return 'gemma'
        elif 'bert' in model_type:
            return 'bert'

        # Check model name as fallback
        name_lower = self.model_name.lower()
        for arch in ['llama', 'mistral', 'gpt2', 'gemma', 'bert']:
            if arch in name_lower:
                return arch

        # Default to llama-like architecture
        return 'llama'

    def _get_model_info(self) -> str:
        """Get model information string."""
        info = []
        if hasattr(self.config, 'hidden_size'):
            info.append(f"hidden_size={self.config.hidden_size}")
        if hasattr(self.config, 'num_hidden_layers'):
            info.append(f"layers={self.config.num_hidden_layers}")
        if hasattr(self.config, 'num_attention_heads'):
            info.append(f"heads={self.config.num_attention_heads}")
        return ", ".join(info)

    def _get_module_by_path(self, path: str) -> Optional[nn.Module]:
        """Get a module by its path in the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        module = self.model
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            elif hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers."""
        layers_path = self.arch_pattern['layers_path']
        return self._get_module_by_path(layers_path)

    def list_available_components(self) -> Dict[str, ComponentSpec]:
        """
        List all extractable components.

        Returns:
            Dict mapping component names to their specifications
        """
        if self.model is None:
            self.load_model()

        components = {}
        hidden_size = getattr(self.config, 'hidden_size', 768)
        num_heads = getattr(self.config, 'num_attention_heads', 12)
        head_dim = hidden_size // num_heads
        intermediate_size = getattr(self.config, 'intermediate_size', hidden_size * 4)
        num_layers = getattr(self.config, 'num_hidden_layers', 12)
        vocab_size = getattr(self.config, 'vocab_size', 50257)

        # Per-layer components
        for layer_idx in range(num_layers):
            # Self-attention
            components[f"self_attention.{layer_idx}"] = ComponentSpec(
                name=f"self_attention.{layer_idx}",
                component_type="attention",
                module_path=f"{self.arch_pattern['layers_path']}.{layer_idx}.{self.arch_pattern['attention']}",
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                description=f"Self-attention mechanism from layer {layer_idx}"
            )

            # Feed-forward network
            components[f"feed_forward.{layer_idx}"] = ComponentSpec(
                name=f"feed_forward.{layer_idx}",
                component_type="ffn",
                module_path=f"{self.arch_pattern['layers_path']}.{layer_idx}.{self.arch_pattern['ffn']}",
                input_dim=hidden_size,
                output_dim=hidden_size,
                intermediate_dim=intermediate_size,
                description=f"Feed-forward network from layer {layer_idx}"
            )

            # Layer norms
            components[f"input_norm.{layer_idx}"] = ComponentSpec(
                name=f"input_norm.{layer_idx}",
                component_type="norm",
                module_path=f"{self.arch_pattern['layers_path']}.{layer_idx}.{self.arch_pattern['input_norm']}",
                input_dim=hidden_size,
                output_dim=hidden_size,
                description=f"Input layer norm from layer {layer_idx}"
            )

            components[f"post_attn_norm.{layer_idx}"] = ComponentSpec(
                name=f"post_attn_norm.{layer_idx}",
                component_type="norm",
                module_path=f"{self.arch_pattern['layers_path']}.{layer_idx}.{self.arch_pattern['post_attn_norm']}",
                input_dim=hidden_size,
                output_dim=hidden_size,
                description=f"Post-attention layer norm from layer {layer_idx}"
            )

        # Global components
        components["embeddings"] = ComponentSpec(
            name="embeddings",
            component_type="embedding",
            module_path=self.arch_pattern['embed'],
            input_dim=vocab_size,
            output_dim=hidden_size,
            description="Token embeddings"
        )

        if 'lm_head' in self.arch_pattern:
            components["lm_head"] = ComponentSpec(
                name="lm_head",
                component_type="linear",
                module_path=self.arch_pattern['lm_head'],
                input_dim=hidden_size,
                output_dim=vocab_size,
                description="Language model head"
            )

        if 'pos_embed' in self.arch_pattern:
            max_pos = getattr(self.config, 'max_position_embeddings', 1024)
            components["position_embeddings"] = ComponentSpec(
                name="position_embeddings",
                component_type="embedding",
                module_path=self.arch_pattern['pos_embed'],
                input_dim=max_pos,
                output_dim=hidden_size,
                description="Position embeddings"
            )

        return components

    def extract_component(
        self,
        component_name: str,
        layer_idx: Optional[int] = None,
        copy_weights: bool = True
    ) -> ExtractedComponent:
        """
        Extract a component from the model.

        Args:
            component_name: Name of component ("self_attention", "feed_forward", etc.)
            layer_idx: Layer index (required for per-layer components)
            copy_weights: If True, deep copy the weights (safe but uses memory)

        Returns:
            ExtractedComponent containing the module and metadata
        """
        if self.model is None:
            self.load_model()

        # Build full component key
        if layer_idx is not None and '.' not in component_name:
            full_name = f"{component_name}.{layer_idx}"
        else:
            full_name = component_name

        # Check cache
        if full_name in self._components_cache:
            return self._components_cache[full_name]

        # Get component spec
        components = self.list_available_components()
        if full_name not in components:
            available = [k for k in components.keys() if component_name in k][:5]
            raise ValueError(
                f"Component '{full_name}' not found. "
                f"Available similar: {available}"
            )

        spec = components[full_name]

        # Extract the actual module
        module = self._get_module_by_path(spec.module_path)
        if module is None:
            raise RuntimeError(f"Could not find module at path: {spec.module_path}")

        # Copy if requested (safer for modification)
        if copy_weights:
            module = copy.deepcopy(module)

        # Get state dict
        state_dict = {k: v.clone() for k, v in module.state_dict().items()}

        # Determine source layer
        source_layer = layer_idx if layer_idx is not None else -1

        extracted = ExtractedComponent(
            module=module,
            spec=spec,
            source_model=self.model_name,
            source_layer=source_layer,
            state_dict=state_dict
        )

        # Cache it
        self._components_cache[full_name] = extracted

        logger.info(f"Extracted {full_name} from {self.model_name}")
        return extracted

    def extract_layer(self, layer_idx: int, copy_weights: bool = True) -> Dict[str, ExtractedComponent]:
        """
        Extract all components from a single layer.

        Args:
            layer_idx: Layer index to extract
            copy_weights: If True, deep copy the weights

        Returns:
            Dict of component name to ExtractedComponent
        """
        components = {}
        for comp_type in ['self_attention', 'feed_forward', 'input_norm', 'post_attn_norm']:
            try:
                components[comp_type] = self.extract_component(
                    comp_type, layer_idx=layer_idx, copy_weights=copy_weights
                )
            except Exception as e:
                logger.warning(f"Could not extract {comp_type} from layer {layer_idx}: {e}")

        return components

    def create_adapter(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True
    ) -> nn.Module:
        """
        Create a dimension adapter (linear projection).

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            bias: Whether to include bias

        Returns:
            nn.Linear adapter module
        """
        return nn.Linear(input_dim, output_dim, bias=bias)

    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        if self.config is None:
            raise RuntimeError("Model not loaded")
        return getattr(self.config, 'num_hidden_layers', 12)

    def get_hidden_size(self) -> int:
        """Get the hidden dimension size."""
        if self.config is None:
            raise RuntimeError("Model not loaded")
        return getattr(self.config, 'hidden_size', 768)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded model."""
        if self.model is None:
            return {"status": "not loaded", "model_name": self.model_name}

        return {
            "model_name": self.model_name,
            "architecture": self.arch_type,
            "hidden_size": self.get_hidden_size(),
            "num_layers": self.get_num_layers(),
            "num_heads": getattr(self.config, 'num_attention_heads', None),
            "vocab_size": getattr(self.config, 'vocab_size', None),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": next(self.model.parameters()).device,
            "dtype": next(self.model.parameters()).dtype,
        }


class ComponentTester:
    """
    Test extracted components in isolation.

    Like putting an engine on a test stand - run it outside the car
    to see how it behaves.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def test_attention(
        self,
        attention: ExtractedComponent,
        batch_size: int = 1,
        seq_len: int = 128
    ) -> Dict[str, Any]:
        """
        Test an attention component.

        Args:
            attention: Extracted attention component
            batch_size: Batch size for test input
            seq_len: Sequence length for test input

        Returns:
            Dict with test results
        """
        module = attention.module.to(self.device)
        module.eval()

        hidden_size = attention.spec.input_dim

        # Create test input
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device=self.device,
            dtype=next(module.parameters()).dtype
        )

        # Create attention mask (all ones = attend to everything)
        attention_mask = torch.ones(
            batch_size, 1, seq_len, seq_len,
            device=self.device,
            dtype=hidden_states.dtype
        )

        # Create position ids
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Run forward pass
        with torch.no_grad():
            try:
                # Try different calling conventions
                try:
                    output = module(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
                except TypeError:
                    try:
                        output = module(hidden_states, attention_mask=attention_mask)
                    except TypeError:
                        output = module(hidden_states)

                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]

                return {
                    "success": True,
                    "input_shape": list(hidden_states.shape),
                    "output_shape": list(output.shape),
                    "output_mean": output.mean().item(),
                    "output_std": output.std().item(),
                    "output_range": [output.min().item(), output.max().item()],
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "input_shape": list(hidden_states.shape),
                }

    def test_ffn(
        self,
        ffn: ExtractedComponent,
        batch_size: int = 1,
        seq_len: int = 128
    ) -> Dict[str, Any]:
        """Test a feed-forward network component."""
        module = ffn.module.to(self.device)
        module.eval()

        hidden_size = ffn.spec.input_dim

        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device=self.device,
            dtype=next(module.parameters()).dtype
        )

        with torch.no_grad():
            try:
                output = module(hidden_states)

                if isinstance(output, tuple):
                    output = output[0]

                return {
                    "success": True,
                    "input_shape": list(hidden_states.shape),
                    "output_shape": list(output.shape),
                    "output_mean": output.mean().item(),
                    "output_std": output.std().item(),
                    "expansion_ratio": ffn.spec.intermediate_dim / ffn.spec.input_dim if ffn.spec.intermediate_dim else None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

    def compare_components(
        self,
        comp_a: ExtractedComponent,
        comp_b: ExtractedComponent,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compare two components (e.g., attention from different models).

        Args:
            comp_a: First component
            comp_b: Second component
            num_samples: Number of random inputs to test

        Returns:
            Comparison statistics
        """
        if comp_a.spec.input_dim != comp_b.spec.input_dim:
            return {
                "comparable": False,
                "reason": f"Dimension mismatch: {comp_a.spec.input_dim} vs {comp_b.spec.input_dim}"
            }

        module_a = comp_a.module.to(self.device).eval()
        module_b = comp_b.module.to(self.device).eval()

        hidden_size = comp_a.spec.input_dim

        cosine_sims = []
        output_diffs = []

        with torch.no_grad():
            for _ in range(num_samples):
                x = torch.randn(1, 32, hidden_size, device=self.device, dtype=torch.float32)

                try:
                    out_a = module_a(x)
                    out_b = module_b(x)

                    if isinstance(out_a, tuple):
                        out_a = out_a[0]
                    if isinstance(out_b, tuple):
                        out_b = out_b[0]

                    # Cosine similarity
                    cos_sim = nn.functional.cosine_similarity(
                        out_a.flatten(), out_b.flatten(), dim=0
                    ).item()
                    cosine_sims.append(cos_sim)

                    # L2 difference
                    diff = (out_a - out_b).norm().item()
                    output_diffs.append(diff)

                except Exception:
                    continue

        if not cosine_sims:
            return {"comparable": False, "reason": "Could not compute any comparisons"}

        return {
            "comparable": True,
            "cosine_similarity_mean": sum(cosine_sims) / len(cosine_sims),
            "cosine_similarity_std": torch.tensor(cosine_sims).std().item(),
            "l2_diff_mean": sum(output_diffs) / len(output_diffs),
            "l2_diff_std": torch.tensor(output_diffs).std().item(),
            "num_samples": len(cosine_sims),
            "component_a": f"{comp_a.source_model}:{comp_a.spec.name}",
            "component_b": f"{comp_b.source_model}:{comp_b.spec.name}",
        }
