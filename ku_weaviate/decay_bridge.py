# ku_weaviate/decay_bridge.py

"""
Decay Bridge: Knowledge Universe → Weaviate Boost.decay

Translates KU's platform-aware half-lives into Weaviate Boost.decay
scale parameters. This is the mathematical glue between the two layers.

KU half-life (days) → Weaviate scale (duration string)

The scale parameter in Weaviate's Boost.decay represents the time
at which the decay function reaches 0.5 (half the boost). This maps
directly to KU's half-life concept.

Formula: scale = f"{half_life_days}d"
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# KU platform half-lives (days) — mirrors src/scoring/decay_engine.py
# Source of truth: https://api.knowledgeuniverse.tech
KU_HALF_LIVES: dict[str, int] = {
    "huggingface":     120,    # ML moves fastest
    "github":          180,    # Code goes stale with dependencies
    "podcast":         180,    # Audio content ages quickly
    "common_crawl":     90,    # Web snapshots — very fast
    "youtube":         270,    # Tutorials date with library releases
    "stackoverflow":   365,    # Answers age with framework versions
    "kaggle":          365,    # Competition context becomes irrelevant
    "wikipedia":      1460,    # Actively maintained, slow to decay
    "arxiv":          1095,    # Research papers have long shelf life
    "mit_ocw":        1095,    # Academic courses revised on cycles
    "openlibrary":    1825,    # Books revised infrequently
    "libgen":         1825,    # Books — very long half-life
    "documentation":   180,    # Docs age with software releases
    "paperswithcode":  365,    # Papers + code
    "crossref":       1095,    # Academic publications
    "distill":        1095,    # High-quality interactive ML research
}

DEFAULT_HALF_LIFE = 365  # fallback for unknown platforms

# Domain velocity labels from KU → recommended Boost depth
# "hypersonic" = LLM releases, 7-day half-life → shallow depth (fast churn)
# "frozen" = HTTP spec, 5-year half-life → deep depth (stable)
VELOCITY_TO_DEPTH: dict[str, int] = {
    "hypersonic": 10,    # Only look at very recent, short list
    "fast":       25,
    "moderate":   50,
    "stable":    100,
    "frozen":    200,
    "unknown":    50,    # safe default
}

# Boost curves from Weaviate — which to use per domain velocity
class BoostCurve(str, Enum):
    GAUSSIAN    = "gaussian"     # smooth falloff — good for stable domains
    LINEAR      = "linear"       # predictable falloff
    EXPONENTIAL = "exponential"  # aggressive recent bias — good for fast domains


VELOCITY_TO_CURVE: dict[str, BoostCurve] = {
    "hypersonic": BoostCurve.EXPONENTIAL,
    "fast":       BoostCurve.EXPONENTIAL,
    "moderate":   BoostCurve.GAUSSIAN,
    "stable":     BoostCurve.GAUSSIAN,
    "frozen":     BoostCurve.LINEAR,
    "unknown":    BoostCurve.GAUSSIAN,
}


@dataclass
class WeaviateBoostParams:
    """
    Parameters for Weaviate Boost.decay TimeDecay condition.
    
    These are computed from KU platform metadata and domain velocity,
    then passed directly to Weaviate's query builder.
    """
    scale: str           # e.g. "180d" — half-life as duration string
    depth: int           # how many extra candidates to fetch for re-scoring
    curve: BoostCurve    # decay curve shape
    platform: str        # source platform
    half_life_days: int  # raw half-life for logging/audit
    velocity_label: str  # domain velocity from KU


class DecayBridge:
    """
    Translates Knowledge Universe decay metadata into Weaviate Boost parameters.
    
    This is the mathematical bridge between the two layers.
    KU provides the domain intelligence; Weaviate applies it at retrieval time.
    
    Usage:
        bridge = DecayBridge()
        params = bridge.get_boost_params("github", velocity_label="fast")
        # Use params.scale and params.depth in Weaviate Boost.decay()
    """

    def get_boost_params(
        self,
        platform: str,
        velocity_label: str = "unknown",
        custom_half_life_days: Optional[int] = None,
    ) -> WeaviateBoostParams:
        """
        Get Weaviate Boost.decay parameters for a given platform + velocity.
        
        Args:
            platform: KU source platform (e.g. "github", "arxiv")
            velocity_label: Domain velocity from KU /v1/discover response
                           ("hypersonic" | "fast" | "moderate" | "stable" | "frozen")
            custom_half_life_days: Override half-life (for regulated pipelines
                                   that need tighter governance)
        
        Returns:
            WeaviateBoostParams ready for Weaviate Boost.decay()
        """
        half_life = (
            custom_half_life_days
            or KU_HALF_LIVES.get(platform, DEFAULT_HALF_LIFE)
        )

        return WeaviateBoostParams(
            scale=f"{half_life}d",
            depth=VELOCITY_TO_DEPTH.get(velocity_label, 50),
            curve=VELOCITY_TO_CURVE.get(velocity_label, BoostCurve.GAUSSIAN),
            platform=platform,
            half_life_days=half_life,
            velocity_label=velocity_label,
        )

    def get_blended_params(
        self,
        platforms: list[str],
        velocity_label: str = "unknown",
    ) -> WeaviateBoostParams:
        """
        For heterogeneous collections (mixed platforms), compute a
        weighted-average scale. This addresses Siddarth's feedback in the
        Weaviate PR: per-object scale for mixed collections.
        
        Uses the minimum half-life (most conservative/aggressive decay)
        to protect against stale content from fast-moving platforms
        contaminating results from stable platforms.
        """
        half_lives = [
            KU_HALF_LIVES.get(p, DEFAULT_HALF_LIFE) for p in platforms
        ]
        # Conservative: use minimum half-life across all platforms
        # This ensures the fastest-changing platform sets the decay rate
        min_half_life = min(half_lives) if half_lives else DEFAULT_HALF_LIFE

        return WeaviateBoostParams(
            scale=f"{min_half_life}d",
            depth=VELOCITY_TO_DEPTH.get(velocity_label, 50),
            curve=VELOCITY_TO_CURVE.get(velocity_label, BoostCurve.GAUSSIAN),
            platform=f"blended({','.join(platforms)})",
            half_life_days=min_half_life,
            velocity_label=velocity_label,
        )

    def scale_for_platform(self, platform: str) -> str:
        """Quick accessor — returns just the scale string."""
        hl = KU_HALF_LIVES.get(platform, DEFAULT_HALF_LIFE)
        return f"{hl}d"