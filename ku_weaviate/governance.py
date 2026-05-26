# ku_weaviate/governance.py

"""
Post-Retrieval Governance Layer

This is what makes the two-layer architecture necessary for regulated
use cases. Weaviate Boost.decay() soft-ranks results. This layer
applies hard gates — binary block/pass decisions — before content
reaches the LLM.

Use case: A stale FDA guideline must NEVER reach the LLM, not just
rank lower. This cannot be achieved with soft-ranking alone.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

KU_BASE_URL = "https://api.knowledgeuniverse.tech"


@dataclass
class GovernanceResult:
    """Result of governance check for a single document."""
    url: str
    passed: bool
    decay_score: float
    decay_label: str           # fresh | aging | stale | decayed | unknown
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
    Knowledge Universe post-retrieval governance.
    
    Applies hard gates to Weaviate results before they reach the LLM.
    Integrates with KU's /v1/knowledge-audit endpoint.
    
    Two gate types:
    1. Decay gate:     decay_score > threshold → BLOCKED
    2. Retraction gate: retracted == True → BLOCKED (always, regardless of threshold)
    
    Usage:
        governor = GovernanceLayer(api_key="ku_test_...")
        report = await governor.audit(urls=weaviate_result_urls)
        clean_urls = report.passed_urls  # safe to send to LLM
    """

    def __init__(
        self,
        api_key: str,
        decay_threshold: float = 0.40,
        retraction_check: bool = True,
        base_url: str = KU_BASE_URL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.decay_threshold = decay_threshold
        self.retraction_check = retraction_check
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def audit(
        self,
        urls: list[str],
        domain_velocity: Optional[str] = None,
    ) -> GovernanceReport:
        """
        Run governance audit on a list of URLs from Weaviate results.
        
        Calls KU /v1/knowledge-audit and applies hard gates.
        
        Args:
            urls: List of document URLs from Weaviate search results
            domain_velocity: From KU knowledge_velocity.velocity_label
                           Used to dynamically adjust threshold for
                           fast-moving domains.
        
        Returns:
            GovernanceReport with passed/blocked lists and full audit trail
        """
        import time
        start = time.perf_counter()

        # Adjust threshold based on domain velocity
        effective_threshold = self._effective_threshold(domain_velocity)

        # Call KU knowledge-audit
        audit_data = await self._call_ku_audit(urls)
        if not audit_data:
            # Fail-safe: if KU is unreachable, block everything
            # This is the correct behavior for regulated pipelines
            logger.error("KU audit unreachable — blocking all results (fail-safe)")
            return self._fail_safe_report(urls, time.perf_counter() - start)

        # Apply gates to each result
        results: list[GovernanceResult] = []
        passed_urls: list[str] = []
        blocked_urls: list[str] = []

        dist = audit_data.get("freshness_distribution", {})
        recs = audit_data.get("recommendations", [])

        # KU audit returns aggregate data — we need per-URL results
        # Re-call decay engine logic locally for per-URL decisions
        # (KU audit gives us platform detection + date extraction for free)
        per_url_results = await self._per_url_audit(urls, effective_threshold)

        for result in per_url_results:
            results.append(result)
            if result.passed:
                passed_urls.append(result.url)
            else:
                blocked_urls.append(result.url)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        max_decay_passed = max(
            (r.decay_score for r in results if r.passed), default=0.0
        )

        report = GovernanceReport(
            total_checked=len(urls),
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

    async def _per_url_audit(
        self,
        urls: list[str],
        threshold: float,
    ) -> list[GovernanceResult]:
        """Audit each URL individually via KU, protected by a concurrency limiter."""
        
        # The Bouncer: Only allow 3 concurrent requests to your API at once
        semaphore = asyncio.Semaphore(3)

        async def bounded_audit(client: httpx.AsyncClient, url: str):
            async with semaphore:
                # Add a tiny 100ms delay to give your server breathing room
                await asyncio.sleep(0.1) 
                return await self._audit_single(client, url, threshold)

        async with httpx.AsyncClient(
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        ) as client:
            # We now use the bounded_audit instead of hitting the server raw
            tasks = [bounded_audit(client, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        clean: list[GovernanceResult] = []
        for url, result in zip(urls, results):
            if isinstance(result, GovernanceResult):
                clean.append(result)
            else:
                # Exception — conservative: mark as blocked with unknown decay
                logger.warning(f"Audit failed for {url}: {result}")
                clean.append(GovernanceResult(
                    url=url,
                    passed=False,
                    decay_score=0.5,
                    decay_label="unknown",
                    retracted=False,
                    block_reason="audit_error",
                    age_days=None,
                    platform="unknown",
                ))
        return clean

    async def _audit_single(
        self,
        client: httpx.AsyncClient,
        url: str,
        threshold: float,
    ) -> GovernanceResult:
        """Audit a single URL through KU."""
        try:
            resp = await client.post(
                f"{self.base_url}/v1/knowledge-audit",
                json={"urls": [url]},
            )
            resp.raise_for_status()
            data = resp.json()

            dist = data.get("freshness_distribution", {})
            total = data.get("total_sources", 1) or 1

            # Infer decay from distribution (single URL audit)
            if dist.get("decayed", 0) > 0:
                decay_score, label = 0.85, "decayed"
            elif dist.get("stale", 0) > 0:
                decay_score, label = 0.65, "stale"
            elif dist.get("aging", 0) > 0:
                decay_score, label = 0.35, "aging"
            elif dist.get("fresh", 0) > 0:
                decay_score, label = 0.15, "fresh"
            else:
                decay_score, label = 0.40, "unknown"

            retracted = False  # KU audit will flag retractions in recommendations
            for rec in data.get("recommendations", []):
                if "retracted" in rec.lower():
                    retracted = True
                    break

            # Apply hard gates
            block_reason = None
            if retracted and self.retraction_check:
                block_reason = "retracted"
            elif decay_score > threshold:
                block_reason = f"decay_score={decay_score:.3f} > threshold={threshold:.2f}"

            return GovernanceResult(
                url=url,
                passed=(block_reason is None),
                decay_score=decay_score,
                decay_label=label,
                retracted=retracted,
                block_reason=block_reason,
                age_days=None,  # KU audit gives aggregate, not per-URL age
                platform=self._infer_platform(url),
            )

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"KU API error {e.response.status_code}: {url}")

    async def _call_ku_audit(self, urls: list[str]) -> Optional[dict]:
        """Call KU /v1/knowledge-audit for aggregate stats."""
        try:
            async with httpx.AsyncClient(
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            ) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/knowledge-audit",
                    json={"urls": urls},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"KU audit failed: {e}")
            return None

    def _effective_threshold(self, velocity: Optional[str]) -> float:
        """
        Dynamically adjust decay threshold based on domain velocity.
        
        Fast-moving domains (LLM releases) need stricter gates:
        - hypersonic: threshold 0.25 (only very fresh content passes)
        - fast:       threshold 0.35
        - moderate:   threshold 0.40 (default)
        - stable:     threshold 0.50
        - frozen:     threshold 0.65 (stable content, relaxed gate)
        """
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
                    f"Governance threshold adjusted: "
                    f"{self.decay_threshold} → {adjusted} "
                    f"(velocity={velocity})"
                )
            return adjusted
        return self.decay_threshold

    def _infer_platform(self, url: str) -> str:
        url_lower = url.lower()
        if "arxiv.org" in url_lower: return "arxiv"
        if "github.com" in url_lower: return "github"
        if "stackoverflow.com" in url_lower: return "stackoverflow"
        if "youtube.com" in url_lower: return "youtube"
        if "huggingface.co" in url_lower: return "huggingface"
        if "kaggle.com" in url_lower: return "kaggle"
        if "wikipedia.org" in url_lower: return "wikipedia"
        return "html"

    def _fail_safe_report(
        self, urls: list[str], elapsed: float
    ) -> GovernanceReport:
        """Conservative fail-safe: block everything if KU is unreachable."""
        results = [
            GovernanceResult(
                url=url,
                passed=False,
                decay_score=1.0,
                decay_label="unknown",
                retracted=False,
                block_reason="ku_unreachable",
                age_days=None,
                platform="unknown",
            )
            for url in urls
        ]
        return GovernanceReport(
            total_checked=len(urls),
            passed=0,
            blocked=len(urls),
            passed_urls=[],
            blocked_urls=urls,
            results=results,
            domain_velocity=None,
            max_decay_in_passed=0.0,
            processing_time_ms=round(elapsed * 1000, 2),
        )

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
                    f"  BLOCKED: {r.url[:60]} "
                    f"reason={r.block_reason}"
                )