"""
ComplianceAgent (v0.3.1): orchestrates deterministic analysis, RAG retrieval,
and a local LLM (Ollama) to produce structured compliance findings.

Workflow per R script:
    1. ``BasicRCodeAnalyzer`` flags candidate violations (deterministic).
    2. For each flag, ``RuleRetriever`` fetches the top-k related rules.
    3. All flags are reviewed by the LLM concurrently (``asyncio.gather`` +
       ``ainvoke``), bounded by a semaphore.
    4. ``PydanticOutputParser`` validates each reply as an ``LLMReview``.
    5. Deterministic metadata (rule_id, line_number, severity) from the
       analyzer flag is merged in to form the final ``ComplianceFinding``.
       The LLM never decides these fields.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from .models.schemas import ComplianceFinding, LLMReview
from .prompts import build_review_prompt
from .rag.retriever import RuleRetriever
from .rag.vector_store import RuleVectorStore
from .tools.code_analyzer import BasicRCodeAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "rules"
DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_CONCURRENCY = 4

# Confidence assigned when the LLM output cannot be parsed and we fall back
# to the deterministic finding.
FALLBACK_CONFIDENCE = 0.5

# Called as progress_callback(completed, total). Invoked once with completed=0
# before LLM calls start, then after each flag finishes.
ProgressCallback = Callable[[int, int], None]


def build_ollama_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """Create a local ChatOllama model (imported lazily so tests need no Ollama)."""
    from langchain_ollama import ChatOllama

    kwargs: Dict[str, Any] = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


class ComplianceAgent:
    """End-to-end R code compliance reviewer."""

    def __init__(
        self,
        rules_dir: Path | str = DEFAULT_RULES_DIR,
        llm: Optional[BaseChatModel] = None,
        retriever: Optional[RuleRetriever] = None,
        analyzer: Optional[BasicRCodeAnalyzer] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = 3,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """
        Args:
            rules_dir: Directory containing JSON compliance rules.
            llm: Chat model (any LangChain Runnable). Defaults to ChatOllama.
            retriever: Rule retriever. Defaults to a ChromaDB-backed ``RuleRetriever``.
            analyzer: Deterministic analyzer. Defaults to ``BasicRCodeAnalyzer``.
            model: Ollama model tag used when ``llm`` is not supplied.
            temperature: Sampling temperature used when ``llm`` is not supplied.
            top_k: Number of related rules to retrieve per flagged snippet.
            max_concurrency: Maximum number of in-flight LLM requests.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self.rules_dir = Path(rules_dir)
        self.top_k = top_k
        self.max_concurrency = max_concurrency

        self.analyzer = analyzer or BasicRCodeAnalyzer(str(self.rules_dir))
        self.retriever = retriever or RuleRetriever(
            vector_store=RuleVectorStore(rules_dir=self.rules_dir), top_k=top_k
        )
        self.llm = llm or build_ollama_llm(model=model, temperature=temperature)

        self.parser = PydanticOutputParser(pydantic_object=LLMReview)
        self.prompt = build_review_prompt().partial(
            format_instructions=self.parser.get_format_instructions()
        )
        self.chain = self.prompt | self.llm | self.parser

        self._rules_by_id: Dict[str, Dict[str, Any]] = {
            r["rule_id"]: r for r in self.analyzer.rules if "rule_id" in r
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def aanalyze(
        self,
        script_path: Path | str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[ComplianceFinding]:
        """
        Run the full agent workflow on an R script, reviewing flags concurrently.

        Returns:
            One ``ComplianceFinding`` per deterministic flag, in analyzer order.
        """
        flags = self.analyzer.analyze_file(str(script_path))
        total = len(flags)
        if progress_callback:
            progress_callback(0, total)
        if not flags:
            return []

        # Retrieval is fast local CPU work; do it up front so only the slow
        # LLM calls run concurrently.
        inputs = [self._build_inputs(flag) for flag in flags]

        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def run_one(flag: Dict[str, Any], payload: Dict[str, Any]) -> ComplianceFinding:
            nonlocal completed
            async with semaphore:
                result = await self._areview(flag, payload)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result

        # gather preserves input order regardless of completion order.
        return list(await asyncio.gather(*(run_one(f, p) for f, p in zip(flags, inputs))))

    def analyze(self, script_path: Path | str) -> List[ComplianceFinding]:
        """Synchronous convenience wrapper around :meth:`aanalyze`.

        Do not call from inside a running event loop; await ``aanalyze`` instead.
        """
        return asyncio.run(self.aanalyze(script_path))

    async def areview_flag(self, flag: Dict[str, Any]) -> ComplianceFinding:
        """Review a single deterministic flag asynchronously."""
        return await self._areview(flag, self._build_inputs(flag))

    def review_flag(self, flag: Dict[str, Any]) -> ComplianceFinding:
        """Review a single deterministic flag synchronously."""
        return asyncio.run(self.areview_flag(flag))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _areview(self, flag: Dict[str, Any], inputs: Dict[str, Any]) -> ComplianceFinding:
        try:
            review: LLMReview = await self.chain.ainvoke(inputs)
        except (OutputParserException, ValidationError) as exc:
            logger.warning(
                "LLM output could not be parsed for %s (line %s): %s",
                flag.get("rule_id"), flag.get("line_number"), exc,
            )
            return self._fallback_finding(flag)
        return self._merge(flag, review)

    def _build_inputs(self, flag: Dict[str, Any]) -> Dict[str, Any]:
        snippet = flag.get("line_content", "")
        flagged_rule_id = flag.get("rule_id")
        flagged_rule = self._strip(self._rules_by_id.get(flagged_rule_id, {
            "rule_id": flagged_rule_id,
            "rule_name": flag.get("rule_name"),
            "severity": flag.get("severity"),
            "recommendation": flag.get("recommendation"),
        }))
        related = [
            r for r in self._retrieve_rules(snippet, flagged_rule_id)
            if r.get("rule_id") != flagged_rule_id
        ]
        return {
            "flagged_rule": json.dumps(flagged_rule, indent=2),
            "line_number": flag.get("line_number", ""),
            "code_snippet": snippet,
            "related_rules": json.dumps(related, indent=2) if related else "[]",
        }

    def _retrieve_rules(self, snippet: str, flagged_rule_id: Optional[str]) -> List[Dict[str, Any]]:
        """Top-k rules for the snippet, always including the deterministically flagged rule."""
        query = snippet.strip() or (flagged_rule_id or "")
        retrieved = self.retriever.retrieve(query, k=self.top_k)
        rules = [r.rule for r in retrieved]

        if flagged_rule_id and flagged_rule_id not in {r.get("rule_id") for r in rules}:
            flagged_rule = self._rules_by_id.get(flagged_rule_id)
            if flagged_rule:
                rules.insert(0, flagged_rule)

        return [self._strip(r) for r in rules]

    @staticmethod
    def _strip(rule: Dict[str, Any]) -> Dict[str, Any]:
        """Remove retrieval bookkeeping keys that the LLM does not need."""
        return {k: v for k, v in rule.items() if k != "source_file"}

    @staticmethod
    def _merge(flag: Dict[str, Any], review: LLMReview) -> ComplianceFinding:
        """Combine deterministic flag metadata with the LLM's narrative fields."""
        return ComplianceFinding(
            rule_id=flag["rule_id"],
            line_number=int(flag["line_number"]),
            severity=flag.get("severity", "UNKNOWN"),
            finding=review.finding,
            evidence=review.evidence,
            recommendation=review.recommendation,
            confidence=review.confidence,
        )

    @staticmethod
    def _fallback_finding(flag: Dict[str, Any]) -> ComplianceFinding:
        """Deterministic finding used when the LLM response is unusable."""
        return ComplianceFinding(
            rule_id=flag.get("rule_id", "UNKNOWN"),
            line_number=int(flag.get("line_number", 1)),
            severity=flag.get("severity", "UNKNOWN"),
            finding=f"{flag.get('rule_name', 'Rule violation')} (deterministic fallback; LLM output unavailable).",
            evidence=flag.get("line_content", ""),
            recommendation=flag.get("recommendation", ""),
            confidence=FALLBACK_CONFIDENCE,
        )
