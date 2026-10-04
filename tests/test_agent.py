"""
Orchestration tests for ComplianceAgent (v0.3.1).

The LLM is mocked (FakeListChatModel or an async RunnableLambda) and the
retriever is a lightweight stub, so these tests never call Ollama or load
embedding models. Async code is driven with ``asyncio.run`` so no pytest
asyncio plugin is required.
"""

import asyncio
import json
import os
import re

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

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


def llm_reply(confidence=0.92, **extra):
    """A valid LLMReview JSON payload, optionally with extra (ignored) keys."""
    payload = {
        "finding": "Treatment arm is hard-coded as a string literal.",
        "evidence": "TRT == 'ACTIVE'",
        "recommendation": "Use mapped treatment codes from the randomization table.",
        "confidence": confidence,
    }
    payload.update(extra)
    return json.dumps(payload)


@pytest.fixture
def analyzer():
    return BasicRCodeAnalyzer(RULES_DIR)


@pytest.fixture
def flags(analyzer):
    return analyzer.analyze_file(VIOLATION_SCRIPT)


@pytest.fixture
def retriever(analyzer):
    comp001 = next(r for r in analyzer.rules if r["rule_id"] == "COMP-001")
    return StubRetriever([comp001])


def make_agent(analyzer, retriever, llm, max_concurrency=4):
    return ComplianceAgent(
        rules_dir=RULES_DIR,
        llm=llm,
        retriever=retriever,
        analyzer=analyzer,
        top_k=3,
        max_concurrency=max_concurrency,
    )


# --------------------------------------------------------------------------- #
# Deterministic metadata injection
# --------------------------------------------------------------------------- #
def test_rule_id_and_line_number_inherited_from_analyzer(analyzer, retriever, flags):
    # The mocked LLM tries to hijack metadata; it must be ignored.
    hostile = llm_reply(rule_id="COMP-999", line_number=999, severity="LOW")
    responses = [f"```json\n{hostile}\n```"] * len(flags)  # fenced JSON must parse too
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=responses))

    findings = asyncio.run(agent.aanalyze(VIOLATION_SCRIPT))

    assert len(findings) == len(flags)
    assert all(isinstance(f, ComplianceFinding) for f in findings)
    for flag, finding in zip(flags, findings):
        assert finding.rule_id == flag["rule_id"]
        assert finding.line_number == flag["line_number"]
        assert finding.severity == flag["severity"]
        assert finding.confidence == pytest.approx(0.92)  # LLM-owned field kept
    assert len(retriever.queries) == len(flags)
    assert all(k == 3 for _, k in retriever.queries)


def test_schema_requires_line_number():
    with pytest.raises(Exception):
        ComplianceFinding(
            rule_id="COMP-001", severity="HIGH", finding="x",
            evidence="y", recommendation="z", confidence=0.5,
        )


def test_format_instructions_exclude_metadata(analyzer, retriever):
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=[]))
    instructions = agent.parser.get_format_instructions()
    assert '"confidence"' in instructions and '"evidence"' in instructions
    assert '"rule_id"' not in instructions
    assert '"line_number"' not in instructions


# --------------------------------------------------------------------------- #
# Async orchestration
# --------------------------------------------------------------------------- #
def make_tracking_llm(state, delay=0.05):
    """Async mock LLM that records peak concurrency and echoes the line number."""

    async def _fake(prompt_value):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(delay)
        state["in_flight"] -= 1
        human = prompt_value.to_messages()[-1].content
        line = re.search(r"FLAGGED CODE SNIPPET \(line (\d+)\)", human).group(1)
        return llm_reply(evidence=f"echo line {line}")

    return RunnableLambda(_fake)


def test_flags_reviewed_concurrently_and_order_preserved(analyzer, retriever, flags):
    state = {"in_flight": 0, "peak": 0}
    agent = make_agent(analyzer, retriever, make_tracking_llm(state), max_concurrency=4)

    findings = asyncio.run(agent.aanalyze(VIOLATION_SCRIPT))

    assert state["peak"] == min(4, len(flags))  # ran in parallel, bounded by semaphore
    # Each LLM reply is paired with its own flag, in analyzer order.
    for flag, finding in zip(flags, findings):
        assert finding.evidence == f"echo line {flag['line_number']}"
        assert finding.line_number == flag["line_number"]


def test_max_concurrency_one_is_sequential(analyzer, retriever):
    state = {"in_flight": 0, "peak": 0}
    agent = make_agent(analyzer, retriever, make_tracking_llm(state, delay=0.01), max_concurrency=1)
    asyncio.run(agent.aanalyze(VIOLATION_SCRIPT))
    assert state["peak"] == 1


def test_progress_callback_reports_each_completion(analyzer, retriever, flags):
    state = {"in_flight": 0, "peak": 0}
    agent = make_agent(analyzer, retriever, make_tracking_llm(state, delay=0.01))
    events = []

    asyncio.run(agent.aanalyze(VIOLATION_SCRIPT, progress_callback=lambda d, t: events.append((d, t))))

    total = len(flags)
    assert events[0] == (0, total)
    assert [d for d, _ in events] == list(range(total + 1))
    assert all(t == total for _, t in events)


def test_sync_analyze_wrapper(analyzer, retriever, flags):
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=[llm_reply()] * len(flags)))
    findings = agent.analyze(VIOLATION_SCRIPT)
    assert [f.line_number for f in findings] == [f["line_number"] for f in flags]


# --------------------------------------------------------------------------- #
# Prompt / context construction
# --------------------------------------------------------------------------- #
def test_prompt_contains_flagged_rule_snippet_and_line(analyzer, retriever, flags):
    flag = next(f for f in flags if f["rule_id"] == "COMP-003")
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=[]))

    messages = agent.prompt.format_messages(**agent._build_inputs(flag))
    system, human = messages[0].content, messages[1].content

    assert "clinical programming compliance reviewer" in system
    assert "Do not output rule_id, line_number, or severity" in system
    flagged_section = human.split("FLAGGED CODE SNIPPET")[0]
    assert '"COMP-003"' in flagged_section  # flagged rule is the primary rule
    assert f"(line {flag['line_number']})" in human
    assert flag["line_content"] in human
    related_section = human.split("RELATED RULES")[1]
    assert '"COMP-001"' in related_section  # retrieved rule as context
    assert '"COMP-003"' not in related_section  # not duplicated


def test_flagged_rule_always_in_context(analyzer, retriever):
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=[]))
    ids = [r["rule_id"] for r in agent._retrieve_rules("set.seed(123)", "COMP-003")]
    assert ids[0] == "COMP-003"
    assert "COMP-001" in ids


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_malformed_llm_output_falls_back_to_deterministic(analyzer, retriever, flags):
    flag = flags[0]
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=["Sorry, no JSON today."]))

    finding = asyncio.run(agent.areview_flag(flag))

    assert finding.rule_id == flag["rule_id"]
    assert finding.line_number == flag["line_number"]
    assert finding.confidence == FALLBACK_CONFIDENCE
    assert "fallback" in finding.finding


def test_clean_script_makes_no_llm_calls(analyzer, retriever):
    agent = make_agent(analyzer, retriever, FakeListChatModel(responses=[]))  # any call would raise
    events = []
    assert asyncio.run(agent.aanalyze(CLEAN_SCRIPT, progress_callback=lambda d, t: events.append((d, t)))) == []
    assert retriever.queries == []
    assert events == [(0, 0)]
