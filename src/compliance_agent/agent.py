"""
ComplianceAgent (v0.3): orchestrates deterministic analysis, RAG retrieval,
and a local LLM (Ollama) to produce structured compliance findings.

Workflow per R script:
    1. ``BasicRCodeAnalyzer`` flags candidate violations (deterministic).
    2. For each flag, ``RuleRetriever`` fetches the top-k relevant rules.
    3. The snippet + rules are sent to the LLM with a strict, grounded prompt.
    4. ``PydanticOutputParser`` validates the reply as a ``ComplianceFinding``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from .models.schemas import ComplianceFinding
from .prompts import build_review_prompt
from .rag.retriever import RuleRetriever
from .rag.vector_store import RuleVectorStore
from .tools.code_analyzer import BasicRCodeAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "rules"
DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_TEMPERATURE = 1.0

# Confidence assigned when the LLM output cannot be parsed and we fall back
# to the deterministic finding.
FALLBACK_CONFIDENCE = 0.5


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
    ) -> None:
        """
        Args:
            rules_dir: Directory containing JSON compliance rules.
            llm: Chat model to use. Defaults to ChatOllama(``model``, ``temperature``).
            retriever: Rule retriever. Defaults to a ChromaDB-backed ``RuleRetriever``.
            analyzer: Deterministic analyzer. Defaults to ``BasicRCodeAnalyzer``.
            model: Ollama model tag used when ``llm`` is not supplied.
            temperature: Sampling temperature used when ``llm`` is not supplied.
            top_k: Number of rules to retrieve per flagged snippet.
        """
        self.rules_dir = Path(rules_dir)
        self.top_k = top_k

        self.analyzer = analyzer or BasicRCodeAnalyzer(str(self.rules_dir))
        self.retriever = retriever or RuleRetriever(
            vector_store=RuleVectorStore(rules_dir=self.rules_dir), top_k=top_k
        )
        self.llm = llm or build_ollama_llm(model=model, temperature=temperature)

        self.parser = PydanticOutputParser(pydantic_object=ComplianceFinding)
        self.prompt = build_review_prompt().partial(
            format_instructions=self.parser.get_format_instructions()
        )
        self.chain = self.prompt | self.llm | self.parser

        # rule_id -> full rule dict, used to guarantee the flagged rule is in context.
        self._rules_by_id: Dict[str, Dict[str, Any]] = {
            r["rule_id"]: r for r in self.analyzer.rules if "rule_id" in r
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, script_path: Path | str) -> List[ComplianceFinding]:
        """
        Run the full agent workflow on an R script.

        Returns:
            One ``ComplianceFinding`` per deterministic flag.
        """
        flags = self.analyzer.analyze_file(str(script_path))
        return [self.review_flag(flag) for flag in flags]

    def review_flag(self, flag: Dict[str, Any]) -> ComplianceFinding:
        """Retrieve context for a single deterministic flag and ask the LLM to review it."""
        snippet = flag.get("line_content", "")
        rules = self._retrieve_rules(snippet, flag.get("rule_id"))

        inputs = {
            "flagged_rule_id": flag.get("rule_id", ""),
            "flagged_rule_name": flag.get("rule_name", ""),
            "line_number": flag.get("line_number", ""),
            "code_snippet": snippet,
            "retrieved_rules": json.dumps(rules, indent=2),
        }

        try:
            finding: ComplianceFinding = self.chain.invoke(inputs)
        except (OutputParserException, ValidationError) as exc:
            logger.warning("LLM output could not be parsed for %s: %s", flag.get("rule_id"), exc)
            return self._fallback_finding(flag)

        allowed_ids = {r.get("rule_id") for r in rules}
        if finding.rule_id not in allowed_ids:
            # Grounding guard: the model cited a rule it was not given.
            logger.warning("LLM returned ungrounded rule_id %r; using fallback.", finding.rule_id)
            return self._fallback_finding(flag)

        return finding

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _retrieve_rules(self, snippet: str, flagged_rule_id: Optional[str]) -> List[Dict[str, Any]]:
        """Top-k rules for the snippet, always including the deterministically flagged rule."""
        query = snippet.strip() or (flagged_rule_id or "")
        retrieved = self.retriever.retrieve(query, k=self.top_k)
        rules = [r.rule for r in retrieved]

        if flagged_rule_id and flagged_rule_id not in {r.get("rule_id") for r in rules}:
            flagged_rule = self._rules_by_id.get(flagged_rule_id)
            if flagged_rule:
                rules.insert(0, flagged_rule)

        # Strip retrieval bookkeeping keys that the LLM does not need.
        return [{k: v for k, v in r.items() if k != "source_file"} for r in rules]

    @staticmethod
    def _fallback_finding(flag: Dict[str, Any]) -> ComplianceFinding:
        """Deterministic finding used when the LLM response is unusable."""
        return ComplianceFinding(
            rule_id=flag.get("rule_id", "UNKNOWN"),
            severity=flag.get("severity", "UNKNOWN"),
            finding=f"{flag.get('rule_name', 'Rule violation')} (deterministic fallback; LLM output unavailable).",
            evidence=f"Line {flag.get('line_number', '?')}: {flag.get('line_content', '')}",
            recommendation=flag.get("recommendation", ""),
            confidence=FALLBACK_CONFIDENCE,
        )
