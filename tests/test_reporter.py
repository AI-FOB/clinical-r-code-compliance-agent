"""Tests for the v0.5.1 reporter: code patching, checking summary, and HTML export."""

import os

import pytest

from src.compliance_agent.models.schemas import ComplianceFinding, LLMReview
from src.compliance_agent.tools.reporter import (
    apply_fixes,
    build_checking_summary,
    generate_reports,
    reconstruct_script,
)

SCRIPT = (
    "# Load data\n"
    "library(dplyr)\n"
    "out <- data %>%\n"
    "  filter(TRT == 'ACTIVE') %>%\n"
    "  mutate(x = 1)\n"
    "set.seed(42)\n"
)


def finding(rule_id, line, fix="", severity="HIGH", confidence=0.9):
    return ComplianceFinding(
        rule_id=rule_id, line_number=line, severity=severity, finding="f",
        evidence="e", recommendation="r", confidence=confidence, suggested_fix=fix,
    )


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "demo.R"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


def test_suggested_fix_strips_markdown_fences():
    review = LLMReview(
        finding="f", evidence="e", recommendation="r", confidence=0.9,
        suggested_fix="```r\nset.seed(trial_seed)\n```",
    )
    assert review.suggested_fix == "set.seed(trial_seed)"
    assert finding("X", 1, fix="`x |> f()`").suggested_fix == "x |> f()"


def test_reconstruct_replaces_lines_and_preserves_indentation(script):
    findings = [
        finding("COMP-001", 4, fix="dplyr::filter(TRT == trt$active) |>"),
        finding("COMP-003", 6, fix="# seed set in setup.R"),
    ]
    lines = reconstruct_script(script, findings, file_level_rule_ids=set()).split("\n")
    assert lines[3] == "  dplyr::filter(TRT == trt$active) |>"  # indentation re-applied
    assert lines[5] == "# seed set in setup.R"
    assert lines[0] == "# Load data"  # untouched lines unchanged
    assert reconstruct_script(script, findings, set()).endswith("\n")


def test_file_level_fix_is_inserted_not_replaced(script):
    header = "# Purpose: demo\n# Inputs: data"
    patch = apply_fixes(script, [finding("COMP-004", 1, fix=header)], file_level_rule_ids={"COMP-004"})
    texts = [l.text for l in patch.updated_lines]
    assert texts[:3] == ["# Purpose: demo", "# Inputs: data", "# Load data"]
    assert patch.lines_inserted == 2 and patch.lines_modified == 0


def test_same_line_conflict_uses_highest_confidence(script):
    findings = [
        finding("COMP-001", 4, fix="a()", confidence=0.7),
        finding("COMP-005", 4, fix="b()", confidence=0.95),
    ]
    patch = apply_fixes(script, findings, set())
    assert patch.updated_lines[3].text == "  b()"
    assert patch.conflict_lines == [4]
    assert patch.superseded == {0: 1}


def test_empty_fix_leaves_script_unchanged(script):
    assert reconstruct_script(script, [finding("COMP-003", 6)], set()) == SCRIPT


def test_checking_summary(script):
    findings = [
        finding("COMP-001", 4, fix="x()", severity="HIGH"),
        finding("COMP-005", 4, severity="LOW", confidence=0.5),
        finding("COMP-003", 6, fix="y()", severity="MEDIUM"),
    ]
    s = build_checking_summary(findings, [script], file_level_rule_ids=set())
    assert s["total_files_scanned"] == 1
    assert s["total_lines_evaluated"] == 6
    assert s["total_findings"] == 3
    assert s["severity_counts"] == {"HIGH": 1, "MEDIUM": 1, "LOW": 1}
    assert s["lines_with_findings"] == 2
    assert s["line_compliance_rate"] == pytest.approx(66.7)
    assert s["perfect_compliance_rate"] == 0.0
    assert s["fixes_suggested"] == 2 and s["lines_modified"] == 2
    assert s["low_confidence_findings"] == 1
    assert s["overall_status"] == "FAIL"


def test_generate_reports_html_structure(script, tmp_path):
    findings = [finding("COMP-001", 4, fix="dplyr::filter(TRT == trt$active) |>")]
    excel, html, updated = generate_reports(findings, str(script), str(tmp_path / "out"))
    assert all(os.path.exists(p) for p in (excel, html, updated))
    content = open(html, encoding="utf-8").read()
    for marker in ('id="summary"', 'id="findings"', 'id="updated-code"', 'class="diff-line del"',
                   'class="diff-line add"', '<pre class="code"><code>'):
        assert marker in content
    assert "&lt;-" in content  # R assignment is HTML-escaped
    assert "trt$active" in open(updated, encoding="utf-8").read()
