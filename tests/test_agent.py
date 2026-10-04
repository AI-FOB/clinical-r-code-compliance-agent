"""
Orchestration tests for ComplianceAgent (v0.3).

The LLM is mocked with LangChain's FakeListChatModel and the retriever with a
lightweight stub, so these tests never call Ollama or load embedding models.
"""

import json
import os

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.compliance_agent.agent import FALLBACK_CONFIDENCE, ComplianceAgent
from src.compliance_agent.models.schemas import ComplianceFinding
from src.compliance_agent.rag.retriever import RetrievedRule
from src.compliance_agent.tools.code_analyzer import BasicRCodeAnalyzer

ROOT = os.path.join(os.path.dirname(__file__), "..")
RULES_DIR = os.path.join(ROOT, "knowledge", "rules")
VIOLATION_SCRIPT = os.path.join(ROOT, "examples", "input", "analysis_script_1.R")
CLEAN_SCRIPT = os.path.join(ROOT, "examples", "input", "clean_script.R")


class StubRetriever:
    """Returns a fixed rule list and records every query it receives."""

    def __init__(self, rules):
        self.rules = rules
        self.queries = []

    def retrieve(self, code_snippet, k=None):
        self.queries.append((code_snippet, k))
        return [
            RetrievedRule(
                rule_id=r["rule_id"],
                rule_name=r["rule_name"],
                severity=r["severity"],
                source_file=f"{r['rule_id']}.json",
                score=1.0,
                rule=r,
            )
            for r in self.rules[: k or 3]
        ]


def llm_reply(rule_id="COMP-001", severity="HIGH", confidence=0.92):
    return json.dumps(
        {
            "rule_id": rule_id,
            "severity": severity,
            "finding": "Treatment arm is hard-coded as a string literal.",
            "evidence": "Line 10: TRT == 'ACTIVE'",
            "recommendation": "Use mapped treatment codes from the randomization table.",
            "confidence": confidence,
        }
    )


@pytest.fixture
def analyzer():
    return BasicRCodeAnalyzer(RULES_DIR)


@pytest.fixture
def retriever(analyzer):
    comp001 = next(r for r in analyzer.rules if r["rule_id"] == "COMP-001")
    return StubRetriever([comp001])


def make_agent(analyzer, retriever, responses):
    llm = FakeListChatModel(responses=responses)
    return ComplianceAgent(rules_dir=RULES_DIR, llm=llm, retriever=retriever, analyzer=analyzer, top_k=3)


def test_pipeline_produces_structured_finding_per_flag(analyzer, retriever):
    n_flags = len(analyzer.analyze_file(VIOLATION_SCRIPT))
    # Markdown-fenced JSON is a common local-model habit; the parser must cope.
    responses = [f"```json\n{llm_reply()}\n```"] * n_flags
    agent = make_agent(analyzer, retriever, responses)

    findings = agent.analyze(VIOLATION_SCRIPT)

    assert len(findings) == n_flags
    assert all(isinstance(f, ComplianceFinding) for f in findings)
    assert len(retriever.queries) == n_flags  # retriever called once per flag
    assert all(k == 3 for _, k in retriever.queries)
    first = findings[0]
    assert first.rule_id == "COMP-001"
    assert first.severity == "HIGH"
    assert first.confidence == pytest.approx(0.92)


def test_prompt_contains_snippet_rules_and_format_instructions(analyzer, retriever):
    flag = next(f for f in analyzer.analyze_file(VIOLATION_SCRIPT) if f["rule_id"] == "COMP-001")
    agent = make_agent(analyzer, retriever, [llm_reply()])

    rules = agent._retrieve_rules(flag["line_content"], flag["rule_id"])
    messages = agent.prompt.format_messages(
        flagged_rule_id=flag["rule_id"],
        flagged_rule_name=flag["rule_name"],
        line_number=flag["line_number"],
        code_snippet=flag["line_content"],
        retrieved_rules=json.dumps(rules, indent=2),
    )
    system, human = messages[0].content, messages[1].content

    assert "clinical programming compliance reviewer" in system
    assert "confidence" in system and "rule_id" in system  # schema injected
    assert flag["line_content"] in human
    assert "COMP-001" in human


def test_flagged_rule_always_in_context(analyzer, retriever):
    # Stub only knows COMP-001; a COMP-003 flag must still get COMP-003 in context.
    agent = make_agent(analyzer, retriever, [])
    rules = agent._retrieve_rules("set.seed(123)", "COMP-003")
    ids = [r["rule_id"] for r in rules]
    assert ids[0] == "COMP-003"
    assert "COMP-001" in ids


def test_malformed_llm_output_falls_back_to_deterministic(analyzer, retriever):
    flag = analyzer.analyze_file(VIOLATION_SCRIPT)[0]
    agent = make_agent(analyzer, retriever, ["Sorry, I cannot produce JSON right now."])

    finding = agent.review_flag(flag)

    assert finding.rule_id == flag["rule_id"]
    assert finding.confidence == FALLBACK_CONFIDENCE
    assert "fallback" in finding.finding


def test_ungrounded_rule_id_falls_back(analyzer, retriever):
    flag = next(f for f in analyzer.analyze_file(VIOLATION_SCRIPT) if f["rule_id"] == "COMP-001")
    agent = make_agent(analyzer, retriever, [llm_reply(rule_id="COMP-999")])

    finding = agent.review_flag(flag)

    assert finding.rule_id == "COMP-001"
    assert finding.confidence == FALLBACK_CONFIDENCE


def test_clean_script_makes_no_llm_calls(analyzer, retriever):
    agent = make_agent(analyzer, retriever, [])  # any LLM call would raise
    assert agent.analyze(CLEAN_SCRIPT) == []
    assert retriever.queries == []
