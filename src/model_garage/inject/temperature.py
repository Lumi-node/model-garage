"""
Temperature Debate - Zero-cost creativity dial.

Like adjusting the fuel mixture - same engine, different performance.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import random


class TemperatureDebate:
    """
    Base class for temperature-based debate strategies.

    Creates two perspectives:
    - Conservative: low temperature, focused distribution
    - Exploratory: high temperature, diverse distribution

    Then reconciles them for creative yet coherent output.
    """

    def __init__(
        self,
        conservative_temp: float = 0.5,
        exploratory_temp: float = 1.2,
    ):
        self.conservative_temp = conservative_temp
        self.exploratory_temp = exploratory_temp

    def get_perspectives(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create conservative and exploratory probability distributions.

        Args:
            logits: Raw logits [batch, vocab_size]

        Returns:
            (probs_conservative, probs_exploratory)
        """
        probs_cons = F.softmax(logits / self.conservative_temp, dim=-1)
        probs_wild = F.softmax(logits / self.exploratory_temp, dim=-1)
        return probs_cons, probs_wild

    def debate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply debate and return blended probabilities.

        Override in subclasses for specific strategies.
        """
        raise NotImplementedError("Subclasses must implement debate()")


class RandomSwitchDebate(TemperatureDebate):
    """
    Randomly switch between conservative and exploratory perspectives.

    Simple and effective - each token either follows the safe path
    or takes a creative leap.

    Recommended: wild_probability = 0.2-0.3
    """

    def __init__(
        self,
        conservative_temp: float = 0.5,
        exploratory_temp: float = 1.2,
        wild_probability: float = 0.2
    ):
        super().__init__(conservative_temp, exploratory_temp)
        self.wild_probability = wild_probability

    def debate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Randomly choose between perspectives for this token.
        """
        probs_cons, probs_wild = self.get_perspectives(logits)

        use_wild = random.random() < self.wild_probability

        if use_wild:
            chosen = probs_wild
            choice = "exploratory"
        else:
            chosen = probs_cons
            choice = "conservative"

        info = {
            "strategy": "random_switch",
            "choice": choice,
            "wild_probability": self.wild_probability,
        }

        return chosen, info


class FilteredBlendDebate(TemperatureDebate):
    """
    Blend perspectives, but filter out unreasonable wild tokens.

    Only allows wild tokens that have some probability in the
    conservative distribution - prevents nonsense.

    Recommended: blend_weight = 0.1-0.2, min_conservative_prob = 0.001
    """

    def __init__(
        self,
        conservative_temp: float = 0.5,
        exploratory_temp: float = 1.2,
        blend_weight: float = 0.15,
        min_conservative_prob: float = 0.001
    ):
        super().__init__(conservative_temp, exploratory_temp)
        self.blend_weight = blend_weight
        self.min_conservative_prob = min_conservative_prob

    def debate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Blend perspectives with filtering.
        """
        probs_cons, probs_wild = self.get_perspectives(logits)

        # Filter: only blend wild tokens that aren't crazy
        valid_mask = (probs_cons > self.min_conservative_prob).float()
        masked_wild = probs_wild * valid_mask
        masked_wild = masked_wild / (masked_wild.sum(dim=-1, keepdim=True) + 1e-10)

        # Blend
        blended = (1 - self.blend_weight) * probs_cons + self.blend_weight * masked_wild

        # Renormalize
        blended = blended / (blended.sum(dim=-1, keepdim=True) + 1e-10)

        info = {
            "strategy": "filtered_blend",
            "blend_weight": self.blend_weight,
            "valid_wild_tokens": valid_mask.sum().item(),
        }

        return blended, info


class AdaptiveDebate(TemperatureDebate):
    """
    Adapt wild probability based on context entropy.

    High entropy context -> use more conservative (already uncertain)
    Low entropy context -> can afford to be wilder
    """

    def __init__(
        self,
        conservative_temp: float = 0.5,
        exploratory_temp: float = 1.2,
        base_wild_prob: float = 0.2,
        entropy_threshold: float = 2.0
    ):
        super().__init__(conservative_temp, exploratory_temp)
        self.base_wild_prob = base_wild_prob
        self.entropy_threshold = entropy_threshold

    def debate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Adapt strategy based on current entropy.
        """
        probs_cons, probs_wild = self.get_perspectives(logits)

        # Compute entropy of conservative distribution
        entropy = -(probs_cons * torch.log(probs_cons + 1e-10)).sum(dim=-1).mean().item()

        # High entropy = be more conservative
        # Low entropy = can be wilder
        if entropy > self.entropy_threshold:
            wild_prob = self.base_wild_prob * 0.5  # Reduce wildness
        else:
            wild_prob = self.base_wild_prob * 1.5  # Increase wildness

        wild_prob = min(wild_prob, 0.5)  # Cap at 50%

        # Apply random switch with adapted probability
        if random.random() < wild_prob:
            chosen = probs_wild
            choice = "exploratory"
        else:
            chosen = probs_cons
            choice = "conservative"

        info = {
            "strategy": "adaptive",
            "entropy": entropy,
            "adapted_wild_prob": wild_prob,
            "choice": choice,
        }

        return chosen, info


# Convenience function for quick use

def debate_sample(
    logits: torch.Tensor,
    strategy: str = "random_switch",
    **kwargs
) -> torch.Tensor:
    """
    Quick helper to apply temperature debate and sample.

    Args:
        logits: Raw logits [batch, vocab_size]
        strategy: "random_switch", "filtered_blend", or "adaptive"
        **kwargs: Strategy-specific parameters

    Returns:
        Sampled token ids [batch, 1]
    """
    if strategy == "random_switch":
        debate = RandomSwitchDebate(**kwargs)
    elif strategy == "filtered_blend":
        debate = FilteredBlendDebate(**kwargs)
    elif strategy == "adaptive":
        debate = AdaptiveDebate(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    probs, _ = debate.debate(logits)
    return torch.multinomial(probs, num_samples=1)
