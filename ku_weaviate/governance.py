"""
Post-Retrieval Governance Layer

Uses decay scores already stored in Weaviate during ingest.
No external API call at query time — governance runs on pre-computed data.

Two gate types:
1. Decay gate:  decay_score > threshold → BLOCKED
2. Platform gate: adjusts threshold per platform velocity
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Platform-aware half-lives (days) — mirrors KU decay engine
KU_HALF_LIVES: dict[str, int] = {
    "huggingface":    120,
    "github":         180,
    "podcast":        180,
    "common_crawl":    90,
    "youtube":        270,
    "stackoverflow":  365,
    "kaggle":         365,
    "wikipedia":     1460,
    "arxiv":         1095,
    "mit_ocw":       1095,
    "openlibrary":   1825,
    "documentation":  180,
    "paperswithcode": 365,
    "crossref":      1095,
    "semantic_scholar": 1095,
    "distill":       1095,
}

DEFAULT_HALF_LIFE = 365


@dataclass
class GovernanceResult:
    """Result of governance check for a single document."""
    url: str
    passed: bool
    decay_score: float
    decay_label: str
    retracted: bool
    block_reason: Optional[str]
    age_days: Optional[int]
    platform: str
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def freshness(self) -> float:
        return round(1.0 - self.decay_score, 3)


@dataclass
class GovernanceReport:
    """Full governance report for a batch of documents."""
    total_checked: int
    passed: int
    blocked: int
    passed_urls: list[str]
    blocked_urls: list[str]
    results: list[GovernanceResult]
    domain_velocity: Optional[str]
    max_decay_in_passed: float
    processing_time_ms: float

    @property
    def block_rate(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return round(self.blocked / self.total_checked, 3)


class GovernanceLayer:
    """
    Post-retrieval governance using decay scores stored in Weaviate.

    No external API calls at query time. Governance runs on data
    that was pre-computed by KU during ingest and stored in Weaviate.

    This is faster, more reliable, and works offline.
    """

    def __init__(
        self,
        api_key: str = "",           # kept for API compatibility
        decay_threshold: float = 0.40,
        retraction_check: bool = True,
        base_url: str = "",          # kept for API compatibility
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.decay_threshold = decay_threshold
        self.retraction_check = retraction_check

    async def audit_from_weaviate_docs(
        self,
        docs: list[dict],
        domain_velocity: Optional[str] = None,
    ) -> GovernanceReport:
        """
        Run governance on documents already retrieved from Weaviate.
        Uses decay_score stored in the document — no external API call.

        Args:
            docs: List of dicts with url, decay_score, platform fields
            domain_velocity: Adjusts threshold dynamically

        Returns:
            GovernanceReport with passed/blocked lists
        """
        start = time.perf_counter()
        effective_threshold = self._effective_threshold(domain_velocity)

        results: list[GovernanceResult] = []
        passed_urls: list[str] = []
        blocked_urls: list[str] = []

        for doc in docs:
            url = doc.get("url", "")
            platform = doc.get("platform", "unknown")
            decay_score = float(doc.get("decay_score") or 0.4)
            decay_label = self._label(decay_score)

            # Apply hard gate
            block_reason = None
            if decay_score > effective_threshold:
                block_reason = (
                    f"decay={decay_score:.3f} > threshold={effective_threshold:.2f}"
                )

            passed = block_reason is None
            result = GovernanceResult(
                url=url,
                passed=passed,
                decay_score=decay_score,
                decay_label=decay_label,
                retracted=False,
                block_reason=block_reason,
                age_days=None,
                platform=platform,
            )
            results.append(result)

            if passed:
                passed_urls.append(url)
            else:
                blocked_urls.append(url)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        max_decay_passed = max(
            (r.decay_score for r in results if r.passed), default=0.0
        )

        report = GovernanceReport(
            total_checked=len(docs),
            passed=len(passed_urls),
            blocked=len(blocked_urls),
            passed_urls=passed_urls,
            blocked_urls=blocked_urls,
            results=results,
            domain_velocity=domain_velocity,
            max_decay_in_passed=round(max_decay_passed, 3),
            processing_time_ms=elapsed_ms,
        )

        self._log_report(report, effective_threshold)
        return report

    # Keep the old audit() method for backwards compatibility
    # but route it through the stored-data path
    async def audit(
        self,
        urls: list[str],
        domain_velocity: Optional[str] = None,
        weaviate_docs: Optional[list[dict]] = None,
    ) -> GovernanceReport:
        """
        Governance audit.
        If weaviate_docs provided: uses stored decay scores (fast, reliable).
        If only urls: builds minimal doc list with platform-inferred defaults.
        """
        if weaviate_docs:
            return await self.audit_from_weaviate_docs(
                weaviate_docs, domain_velocity
            )

        # Fallback: infer platform from URL, use default decay=0.4
        docs = [
            {
                "url": url,
                "platform": self._infer_platform(url),
                "decay_score": 0.4,   # conservative unknown
            }
            for url in urls
        ]
        return await self.audit_from_weaviate_docs(docs, domain_velocity)

    def _effective_threshold(self, velocity: Optional[str]) -> float:
        """Adjust threshold based on domain velocity."""
        velocity_thresholds = {
            "hypersonic": 0.25,
            "fast":       0.35,
            "moderate":   0.40,
            "stable":     0.50,
            "frozen":     0.65,
        }
        if velocity and velocity in velocity_thresholds:
            adjusted = velocity_thresholds[velocity]
            if adjusted != self.decay_threshold:
                logger.info(
                    f"Threshold adjusted: {self.decay_threshold} → {adjusted} "
                    f"(velocity={velocity})"
                )
            return adjusted
        return self.decay_threshold

    def _label(self, decay: float) -> str:
        if decay < 0.25: return "fresh"
        if decay < 0.50: return "aging"
        if decay < 0.75: return "stale"
        return "decayed"

    def _infer_platform(self, url: str) -> str:
        url_lower = url.lower()
        if "arxiv.org" in url_lower:         return "arxiv"
        if "github.com" in url_lower:        return "github"
        if "stackoverflow.com" in url_lower: return "stackoverflow"
        if "youtube.com" in url_lower:       return "youtube"
        if "huggingface.co" in url_lower:    return "huggingface"
        if "kaggle.com" in url_lower:        return "kaggle"
        if "wikipedia.org" in url_lower:     return "wikipedia"
        return "html"

    def _log_report(self, report: GovernanceReport, threshold: float):
        logger.info(
            f"Governance: {report.passed}/{report.total_checked} passed "
            f"(threshold={threshold:.2f}, "
            f"block_rate={report.block_rate:.1%}, "
            f"ms={report.processing_time_ms})"
        )
        for r in report.results:
            if not r.passed:
                logger.info(
                    f"  BLOCKED [{r.platform}] "
                    f"decay={r.decay_score:.2f} → {r.block_reason}"
                )