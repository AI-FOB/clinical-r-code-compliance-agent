"""
Clinical validation reporting (v0.5.1).

Turns a list of compliance findings into actionable deliverables:

* ``reconstruct_script`` - patches the original R script by replacing each
  flagged line with its ``suggested_fix`` (file-level fixes such as a missing
  documentation header are inserted at the top of the file instead).
* ``build_checking_summary`` - execution statistics and severity breakdown.
* ``generate_reports`` - writes an Excel workbook, a patched ``*_updated.R``
  script, and an HTML dashboard (summary, findings with code diffs, and the
  fully updated script).

Findings may be deterministic analyzer dicts or ``ComplianceFinding`` objects.
"""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from jinja2 import BaseLoader, Environment

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[3] / "knowledge" / "rules"
FILE_LEVEL_RULE_TYPE = "file_level_presence"
SEVERITY_ORDER = ["HIGH", "MEDIUM", "LOW"]
LOW_CONFIDENCE_THRESHOLD = 0.7

FindingLike = Union[Dict[str, Any], Any]


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _get(f: FindingLike, key: str, default: Any = None) -> Any:
    return f.get(key, default) if isinstance(f, dict) else getattr(f, key, default)


def _normalize_finding(f: FindingLike, file_name: str) -> Dict[str, Any]:
    """Flatten a dict or ComplianceFinding into a uniform report row."""
    confidence = _get(f, "confidence")
    return {
        "file_name": os.path.basename(file_name),
        "line_number": int(_get(f, "line_number", 1) or 1),
        "rule_id": _get(f, "rule_id", "") or "",
        "severity": str(_get(f, "severity", "UNKNOWN") or "UNKNOWN").upper(),
        "finding": _get(f, "finding") or _get(f, "rule_name", "") or "",
        "evidence": _get(f, "evidence") or _get(f, "line_content", "") or "",
        "recommendation": _get(f, "recommendation", "") or "",
        "confidence": float(confidence) if confidence is not None else None,
        "suggested_fix": (_get(f, "suggested_fix", "") or "").rstrip(),
    }


def load_file_level_rule_ids(rules_dir: Union[str, Path, None] = None) -> Set[str]:
    """Rule IDs whose fix is a file-level insertion rather than a line replacement."""
    rules_path = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    ids: Set[str] = set()
    if not rules_path.is_dir():
        return ids
    for path in rules_path.glob("*.json"):
        try:
            rule = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rule.get("rule_type") == FILE_LEVEL_RULE_TYPE and rule.get("rule_id"):
            ids.add(rule["rule_id"])
    return ids


def _read_script(script_path: Union[str, Path]) -> Tuple[List[str], bool]:
    """Return (lines, had_trailing_newline). Line numbering matches the analyzer."""
    text = Path(script_path).read_text(encoding="utf-8")
    trailing = text.endswith("\n")
    lines = text.split("\n")
    if trailing:
        lines = lines[:-1]
    return lines, trailing


def _indent_like(fix: str, original_line: str) -> str:
    """Re-apply the original line's indentation if the fix omits it."""
    fix_lines = fix.split("\n")
    if fix_lines and fix_lines[0][:1] in (" ", "\t"):
        return fix
    indent = original_line[: len(original_line) - len(original_line.lstrip())]
    if not indent:
        return fix
    return "\n".join(indent + l if l.strip() else l for l in fix_lines)


# --------------------------------------------------------------------------- #
# Code patching
# --------------------------------------------------------------------------- #
@dataclass
class CodeLine:
    text: str
    status: str  # "unchanged" | "modified" | "inserted"
    original_line_number: Optional[int] = None


@dataclass
class PatchResult:
    original_lines: List[str]
    updated_lines: List[CodeLine]
    trailing_newline: bool
    applied_indices: Set[int] = field(default_factory=set)  # findings whose fix was used
    superseded: Dict[int, int] = field(default_factory=dict)  # finding idx -> winning idx
    conflict_lines: List[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        body = "\n".join(l.text for l in self.updated_lines)
        return body + ("\n" if self.trailing_newline else "")

    @property
    def lines_modified(self) -> int:
        return len({l.original_line_number for l in self.updated_lines if l.status == "modified"})

    @property
    def lines_inserted(self) -> int:
        return sum(1 for l in self.updated_lines if l.status == "inserted")


def apply_fixes(
    script_path: Union[str, Path],
    findings: Sequence[FindingLike],
    file_level_rule_ids: Optional[Iterable[str]] = None,
) -> PatchResult:
    """Patch the script with each finding's ``suggested_fix``.

    Rules:
      * Line-level fix: replaces the original line at ``line_number``.
      * File-level fix (e.g. missing header): inserted at the top of the file.
      * Several fixes for the same line: the highest-confidence fix wins (the
        agent asks the LLM to make each fix resolve all co-flagged rules).
      * Empty fixes, or fixes identical to the original line, change nothing.
    """
    file_level = set(file_level_rule_ids) if file_level_rule_ids is not None else load_file_level_rule_ids()
    original, trailing = _read_script(script_path)
    rows = [_normalize_finding(f, str(script_path)) for f in findings]

    candidates: Dict[int, List[int]] = {}
    insertions: List[int] = []
    for idx, row in enumerate(rows):
        if not row["suggested_fix"].strip():
            continue
        if row["rule_id"] in file_level:
            insertions.append(idx)
        elif 1 <= row["line_number"] <= len(original):
            candidates.setdefault(row["line_number"], []).append(idx)

    result = PatchResult(original_lines=original, updated_lines=[], trailing_newline=trailing)

    seen_blocks: Set[str] = set()
    for idx in insertions:
        block = rows[idx]["suggested_fix"]
        if block.strip() in seen_blocks:
            continue
        seen_blocks.add(block.strip())
        result.applied_indices.add(idx)
        result.updated_lines.extend(CodeLine(l, "inserted") for l in block.split("\n"))

    chosen: Dict[int, int] = {}
    for line_no, idxs in candidates.items():
        # max() keeps the first of equal-confidence candidates (analyzer order).
        winner = max(idxs, key=lambda i: rows[i]["confidence"] or 0.0)
        chosen[line_no] = winner
        distinct = {rows[i]["suggested_fix"].strip() for i in idxs}
        if len(distinct) > 1:
            result.conflict_lines.append(line_no)
        for i in idxs:
            if i != winner:
                result.superseded[i] = winner

    for line_no, line in enumerate(original, start=1):
        winner = chosen.get(line_no)
        fix = rows[winner]["suggested_fix"] if winner is not None else None
        if fix is not None and fix.strip() != line.strip():
            result.applied_indices.add(winner)
            new_text = _indent_like(fix, line)
            result.updated_lines.extend(CodeLine(l, "modified", line_no) for l in new_text.split("\n"))
        else:
            result.updated_lines.append(CodeLine(line, "unchanged", line_no))

    return result


def reconstruct_script(
    script_path: Union[str, Path],
    findings: Sequence[FindingLike],
    file_level_rule_ids: Optional[Iterable[str]] = None,
) -> str:
    """Return the full R script with every ``suggested_fix`` applied."""
    return apply_fixes(script_path, findings, file_level_rule_ids).text


# --------------------------------------------------------------------------- #
# Checking summary
# --------------------------------------------------------------------------- #
def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def build_checking_summary(
    findings: Sequence[FindingLike],
    script_paths: Sequence[Union[str, Path]],
    file_level_rule_ids: Optional[Iterable[str]] = None,
    patches: Optional[Dict[str, PatchResult]] = None,
) -> Dict[str, Any]:
    """Execution statistics for one or more scanned scripts.

    ``findings`` may be ComplianceFinding objects, analyzer dicts, or already
    normalised rows; a row's ``file_name`` (basename) links it to its script.
    """
    file_level = set(file_level_rule_ids) if file_level_rule_ids is not None else load_file_level_rule_ids()
    default_file = os.path.basename(str(script_paths[0])) if script_paths else ""
    rows = [
        f if isinstance(f, dict) and "file_name" in f and "suggested_fix" in f
        else _normalize_finding(f, default_file)
        for f in findings
    ]

    total_lines = non_blank = flagged_lines = modified = inserted = conflicts = 0
    files_with_findings = 0
    for path in script_paths:
        name = os.path.basename(str(path))
        lines, _ = _read_script(path)
        total_lines += len(lines)
        non_blank += sum(1 for l in lines if l.strip())
        file_rows = [r for r in rows if r["file_name"] == name]
        if file_rows:
            files_with_findings += 1
        flagged_lines += len({r["line_number"] for r in file_rows if r["rule_id"] not in file_level})
        patch = (patches or {}).get(name) or apply_fixes(path, file_rows, file_level)
        modified += patch.lines_modified
        inserted += patch.lines_inserted
        conflicts += len(patch.conflict_lines)

    files_scanned = len(script_paths)
    total = len(rows)
    sev_counts: Dict[str, int] = {s: 0 for s in SEVERITY_ORDER}
    for r in rows:
        sev_counts[r["severity"]] = sev_counts.get(r["severity"], 0) + 1

    rule_counts: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        entry = rule_counts.setdefault(r["rule_id"], {"count": 0, "severity": r["severity"]})
        entry["count"] += 1

    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    fixes = sum(1 for r in rows if r["suggested_fix"].strip())

    if total == 0:
        status = "PASS"
    elif sev_counts.get("HIGH", 0):
        status = "FAIL"
    else:
        status = "WARN"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": status,
        "total_files_scanned": files_scanned,
        "files_with_findings": files_with_findings,
        "files_fully_compliant": files_scanned - files_with_findings,
        "perfect_compliance_rate": _pct(files_scanned - files_with_findings, files_scanned),
        "total_lines_evaluated": total_lines,
        "non_blank_lines": non_blank,
        "lines_with_findings": flagged_lines,
        "compliant_lines": total_lines - flagged_lines,
        "line_compliance_rate": _pct(total_lines - flagged_lines, total_lines),
        "total_findings": total,
        "severity_counts": sev_counts,
        "severity_percentages": {s: _pct(c, total) for s, c in sev_counts.items()},
        "unique_rules_triggered": len(rule_counts),
        "rule_counts": dict(sorted(rule_counts.items())),
        "file_level_findings": sum(1 for r in rows if r["rule_id"] in file_level),
        "fixes_suggested": fixes,
        "fix_coverage_rate": _pct(fixes, total),
        "lines_modified": modified,
        "lines_inserted": inserted,
        "fix_conflicts": conflicts,
        "average_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "low_confidence_findings": sum(1 for c in confidences if c < LOW_CONFIDENCE_THRESHOLD),
    }


# --------------------------------------------------------------------------- #
# Diff helpers
# --------------------------------------------------------------------------- #
def _inline_diff(old: str, new: str) -> Tuple[List[Tuple[str, bool]], List[Tuple[str, bool]]]:
    """Character-level diff segments: ([(text, removed)], [(text, added)])."""
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    if matcher.ratio() < 0.3:  # mostly rewritten: highlight whole lines
        return [(old, True)], [(new, True)]
    old_seg: List[Tuple[str, bool]] = []
    new_seg: List[Tuple[str, bool]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            old_seg.append((old[i1:i2], False))
            new_seg.append((new[j1:j2], False))
        else:
            if i2 > i1:
                old_seg.append((old[i1:i2], True))
            if j2 > j1:
                new_seg.append((new[j1:j2], True))
    return old_seg, new_seg


def _build_finding_views(
    rows: List[Dict[str, Any]],
    patch: Optional[PatchResult],
    file_level: Set[str],
) -> List[Dict[str, Any]]:
    original = patch.original_lines if patch else []
    views = []
    for idx, row in enumerate(rows):
        view = dict(row)
        is_file_level = row["rule_id"] in file_level
        ln = row["line_number"]
        original_line = original[ln - 1] if 1 <= ln <= len(original) else row["evidence"]
        fix = row["suggested_fix"]
        view["is_file_level"] = is_file_level
        view["original_line"] = original_line

        if not fix.strip():
            view["fix_status"] = "none"
        elif is_file_level:
            view["fix_status"] = "insertion"
        elif patch and idx in patch.superseded:
            view["fix_status"] = "superseded"
            view["superseded_by"] = rows[patch.superseded[idx]]["rule_id"]
        elif fix.strip() == original_line.strip():
            view["fix_status"] = "unchanged"
        else:
            view["fix_status"] = "applied"

        display_fix = fix if is_file_level else _indent_like(fix, original_line)
        if is_file_level:
            view["old_segments"] = []
            view["new_segments"] = [(display_fix, True)]
        elif fix.strip():
            view["old_segments"], view["new_segments"] = _inline_diff(original_line, display_fix)
        else:
            view["old_segments"], view["new_segments"] = [(original_line, False)], []
        views.append(view)
    return views


# --------------------------------------------------------------------------- #
# HTML template
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compliance Report · {{ file_name }}</title>
<meta name="description" content="Clinical R code compliance validation report for {{ file_name }}: checking summary, findings with code diffs, and the fully remediated script.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0b0f1a;--bg-2:#111729;--panel:rgba(22,29,50,.72);--panel-solid:#161d32;--border:rgba(148,163,209,.14);
  --text:#e6e9f5;--muted:#8d95b3;--accent:#7c8cff;--accent-2:#4fd1c5;
  --high:#ff5c7a;--medium:#ffb547;--low:#4cc9f0;--other:#a0a4b8;
  --add-bg:rgba(46,204,138,.12);--add-fg:#5ee6a8;--add-strong:rgba(46,204,138,.32);
  --del-bg:rgba(255,92,122,.10);--del-fg:#ff8fa3;--del-strong:rgba(255,92,122,.32);
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
  --radius:16px;
}
*{box-sizing:border-box}
code,pre,.diff,.loc,.rule,.num,.sec-num,.code-file,.stats-list b,.conf .v{font-variant-ligatures:none;font-feature-settings:"liga" 0,"calt" 0}
html{scroll-behavior:smooth}
body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--text);
  background:radial-gradient(1200px 600px at 10% -10%,rgba(124,140,255,.18),transparent 60%),
             radial-gradient(900px 500px at 110% 10%,rgba(79,209,197,.12),transparent 60%),var(--bg);
  min-height:100vh;line-height:1.55}
.wrap{max-width:1280px;margin:0 auto;padding:40px 28px 80px}
a{color:var(--accent)}
/* Header */
.hero{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-end;justify-content:space-between;margin-bottom:28px;animation:rise .5s ease both}
.brand{display:flex;gap:16px;align-items:center}
.logo{width:52px;height:52px;border-radius:14px;display:grid;place-items:center;font-weight:800;font-size:20px;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));color:#0b0f1a;box-shadow:0 10px 30px rgba(124,140,255,.35)}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{margin:2px 0 0;font-size:30px;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(90deg,#fff,#b9c2ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.meta{display:flex;flex-wrap:wrap;gap:10px}
.chip{font-size:12.5px;padding:7px 12px;border-radius:999px;background:var(--panel);border:1px solid var(--border);color:var(--muted)}
.chip b{color:var(--text);font-weight:600}
.status{font-weight:700;letter-spacing:.06em}
.status.PASS{color:#0b0f1a;background:var(--add-fg);border-color:transparent}
.status.WARN{color:#0b0f1a;background:var(--medium);border-color:transparent}
.status.FAIL{color:#fff;background:var(--high);border-color:transparent;box-shadow:0 0 0 4px rgba(255,92,122,.18)}
nav.toc{position:sticky;top:0;z-index:5;display:flex;gap:6px;padding:10px;margin:0 0 28px;border-radius:14px;
  background:rgba(11,15,26,.75);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid var(--border)}
nav.toc a{color:var(--muted);text-decoration:none;font-size:13.5px;font-weight:600;padding:8px 14px;border-radius:10px;transition:.2s}
nav.toc a:hover{color:var(--text);background:rgba(124,140,255,.14)}
section{margin-bottom:44px;animation:rise .6s ease both}
section:nth-of-type(2){animation-delay:.08s}section:nth-of-type(3){animation-delay:.16s}
.sec-head{display:flex;align-items:baseline;gap:14px;margin-bottom:18px}
.sec-num{font-family:var(--mono);font-size:13px;color:var(--accent);border:1px solid rgba(124,140,255,.4);padding:3px 9px;border-radius:8px}
h2{margin:0;font-size:22px;font-weight:700;letter-spacing:-.01em}
.sec-sub{color:var(--muted);font-size:14px;margin:-8px 0 18px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);box-shadow:0 20px 50px rgba(0,0,0,.25)}
/* KPI grid */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:16px}
.kpi{padding:18px 18px 16px;transition:transform .2s,border-color .2s}
.kpi:hover{transform:translateY(-3px);border-color:rgba(124,140,255,.45)}
.kpi .label{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em}
.kpi .value{font-size:30px;font-weight:800;letter-spacing:-.02em;margin-top:6px}
.kpi .hint{font-size:12.5px;color:var(--muted);margin-top:2px}
.ring{--p:0;--c:var(--accent-2);width:64px;height:64px;border-radius:50%;flex:none;
  background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(148,163,209,.15) 0);display:grid;place-items:center}
.ring::after{content:"";width:48px;height:48px;border-radius:50%;background:var(--panel-solid)}
.kpi.with-ring{display:flex;align-items:center;justify-content:space-between;gap:12px}
.grid-2{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}
@media (max-width:900px){.grid-2{grid-template-columns:1fr}}
.card-pad{padding:22px}
.card-title{font-size:14px;font-weight:700;margin:0 0 14px;color:#cfd5f0}
.stack{display:flex;height:14px;border-radius:999px;overflow:hidden;background:rgba(148,163,209,.12);margin-bottom:18px}
.stack span{display:block;height:100%;transition:width .8s ease}
.sev-rows{display:grid;gap:10px}
.sev-row{display:grid;grid-template-columns:100px 1fr auto;align-items:center;gap:12px;font-size:14px}
.bar{height:8px;border-radius:999px;background:rgba(148,163,209,.12);overflow:hidden}
.bar i{display:block;height:100%;border-radius:999px}
.num{font-family:var(--mono);text-align:right;color:var(--muted);white-space:nowrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}
.HIGH-c{background:var(--high)}.MEDIUM-c{background:var(--medium)}.LOW-c{background:var(--low)}.OTHER-c{background:var(--other)}
.mini-table{width:100%;border-collapse:collapse;font-size:13.5px}
.mini-table th{text-align:left;color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em;padding:0 8px 10px}
.mini-table td{padding:9px 8px;border-top:1px solid var(--border)}
.stats-list{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-top:18px;font-size:13.5px}
.stats-list div{display:flex;justify-content:space-between;border-bottom:1px dashed var(--border);padding-bottom:6px;color:var(--muted)}
.stats-list b{color:var(--text);font-family:var(--mono);font-weight:600}
/* Badges */
.badge{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:700;letter-spacing:.06em;padding:4px 10px;border-radius:999px;white-space:nowrap}
.badge.HIGH{color:var(--high);background:rgba(255,92,122,.12);border:1px solid rgba(255,92,122,.35)}
.badge.MEDIUM{color:var(--medium);background:rgba(255,181,71,.12);border:1px solid rgba(255,181,71,.35)}
.badge.LOW{color:var(--low);background:rgba(76,201,240,.12);border:1px solid rgba(76,201,240,.35)}
.badge.OTHER{color:var(--other);background:rgba(160,164,184,.12);border:1px solid rgba(160,164,184,.35)}
.tag{font-size:11px;font-weight:600;padding:3px 8px;border-radius:6px;background:rgba(148,163,209,.12);color:var(--muted);white-space:nowrap}
.tag.applied{color:var(--add-fg);background:var(--add-bg)}
.tag.insertion{color:var(--accent-2);background:rgba(79,209,197,.12)}
.tag.superseded{color:var(--medium);background:rgba(255,181,71,.12)}
.tag.none{color:var(--del-fg);background:var(--del-bg)}
/* Findings table */
.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.filter{font:600 13px Inter,sans-serif;color:var(--muted);background:var(--panel);border:1px solid var(--border);padding:8px 14px;border-radius:10px;cursor:pointer;transition:.2s}
.filter:hover{color:var(--text);border-color:rgba(124,140,255,.45)}
.filter.active{color:#0b0f1a;background:linear-gradient(135deg,var(--accent),var(--accent-2));border-color:transparent}
.table-wrap{overflow-x:auto}
table.findings{width:100%;border-collapse:separate;border-spacing:0;font-size:14px}
table.findings thead th{position:sticky;top:0;text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  font-weight:700;padding:14px 12px;background:var(--panel-solid);border-bottom:1px solid var(--border);white-space:nowrap}
table.findings tbody td{padding:18px 12px;vertical-align:top;border-bottom:1px solid var(--border)}
table.findings tbody tr{transition:background .2s}
table.findings tbody tr:hover{background:rgba(124,140,255,.05)}
table.findings tbody tr:last-child td{border-bottom:none}
.sev-cell{border-left:3px solid transparent}
tr.HIGH .sev-cell{border-left-color:var(--high)}tr.MEDIUM .sev-cell{border-left-color:var(--medium)}tr.LOW .sev-cell{border-left-color:var(--low)}
.loc{font-family:var(--mono);font-size:13px;color:#cfd5f0;white-space:nowrap}
.loc small{display:block;color:var(--muted);font-size:11.5px;margin-top:3px}
.rule{font-family:var(--mono);font-weight:600;color:#c6ccff;white-space:nowrap}
.finding-text{margin:0 0 8px;color:#dfe3f3}
.rec{margin:0;font-size:13px;color:var(--muted)}
.rec b{color:#b9c2ff;font-weight:600}
.diff{font-family:var(--mono);font-size:12.5px;border-radius:12px;overflow:hidden;border:1px solid var(--border);background:#0a0e18;min-width:280px}
.diff-head{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 12px;font-family:Inter,sans-serif;font-size:11.5px;color:var(--muted);background:rgba(148,163,209,.06);border-bottom:1px solid var(--border)}
.diff-line{display:grid;grid-template-columns:44px 18px 1fr;white-space:pre-wrap;word-break:break-word}
.diff-line .g{color:#5b6488;text-align:right;padding:6px 6px 6px 0;user-select:none}
.diff-line .s{padding:6px 0;user-select:none;font-weight:700}
.diff-line code{padding:6px 12px 6px 0;font-family:inherit}
.diff-line.del{background:var(--del-bg);color:var(--del-fg)}
.diff-line.del .s{color:var(--high)}
.diff-line.del code{text-decoration:line-through;text-decoration-color:rgba(255,143,163,.55)}
.diff-line.del mark{background:var(--del-strong);color:#ffd1da;border-radius:3px}
.diff-line.add{background:var(--add-bg);color:var(--add-fg)}
.diff-line.add .s{color:#2ecc8a}
.diff-line.add mark{background:var(--add-strong);color:#d6ffe9;border-radius:3px}
.diff-line.ctx{color:#aab2d3}
.diff-note{padding:8px 12px;font-family:Inter,sans-serif;font-size:12px;color:var(--muted);border-top:1px solid var(--border)}
.conf{min-width:96px}
.conf .v{font-family:var(--mono);font-weight:600;font-size:13px}
.conf .bar{margin-top:6px}
.empty{padding:40px;text-align:center;color:var(--muted)}
/* Updated code */
.code-head{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--border)}
.dots{display:flex;gap:6px}.dots i{width:11px;height:11px;border-radius:50%;background:#ff5f57}.dots i:nth-child(2){background:#febc2e}.dots i:nth-child(3){background:#28c840}
.code-file{font-family:var(--mono);font-size:13px;color:#cfd5f0;margin-left:12px}
.code-actions{display:flex;gap:10px;align-items:center}
.btn{font:600 13px Inter,sans-serif;color:#0b0f1a;background:linear-gradient(135deg,var(--accent),var(--accent-2));border:0;padding:9px 16px;border-radius:10px;cursor:pointer;transition:transform .15s,box-shadow .2s}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 22px rgba(124,140,255,.35)}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted)}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.legend .lm::before{background:var(--add-strong)}.legend .li::before{background:rgba(79,209,197,.35)}
pre.code{margin:0;max-height:620px;overflow:auto;background:#0a0e18;border-radius:0 0 var(--radius) var(--radius);padding:14px 0;font-size:13px;line-height:1.7}
pre.code code{font-family:var(--mono);display:block;min-width:max-content}
.cl{display:block;padding:0 22px 0 0;border-left:3px solid transparent}
.cl .n{display:inline-block;width:56px;padding-right:16px;text-align:right;color:#4a5275;user-select:none}
.cl .o{display:inline-block;width:56px;padding-right:16px;text-align:right;color:#3a4160;user-select:none;font-size:11px}
.cl.modified{background:var(--add-bg);border-left-color:#2ecc8a;color:#d6ffe9}
.cl.inserted{background:rgba(79,209,197,.10);border-left-color:var(--accent-2);color:#c9fff7}
.cl:hover{background:rgba(124,140,255,.07)}
.notice{padding:14px 18px;font-size:13.5px;color:var(--medium);background:rgba(255,181,71,.08);border-bottom:1px solid var(--border)}
footer{color:var(--muted);font-size:12.5px;text-align:center;margin-top:40px;line-height:1.7}
footer .warn{display:inline-block;padding:10px 16px;border-radius:12px;border:1px solid rgba(255,181,71,.3);background:rgba(255,181,71,.07);color:#ffd59a;margin-bottom:10px}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
@media print{body{background:#fff;color:#111}.panel,nav.toc{backdrop-filter:none}nav.toc,.toolbar,.btn{display:none}pre.code{max-height:none}}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <div class="brand">
    <div class="logo">Rx</div>
    <div>
      <div class="eyebrow">Clinical Validation Report</div>
      <h1>R Code Compliance Report</h1>
    </div>
  </div>
  <div class="meta">
    <span class="chip">File <b>{{ file_name }}</b></span>
    <span class="chip">Mode <b>{{ run.mode }}</b></span>
    {% if run.model %}<span class="chip">Model <b>{{ run.model }}</b></span>{% endif %}
    <span class="chip">Generated <b>{{ summary.generated_at }}</b></span>
    <span class="chip status {{ summary.overall_status }}">{{ summary.overall_status }}</span>
  </div>
</header>

<nav class="toc" aria-label="Report sections">
  <a href="#summary">01 · Checking Summary</a>
  <a href="#findings">02 · Findings &amp; Code Diffs</a>
  <a href="#updated-code">03 · Fully Updated Code</a>
</nav>

<main>
<!-- ============================ SECTION 1 ============================ -->
<section id="summary">
  <div class="sec-head"><span class="sec-num">01</span><h2>Checking Summary</h2></div>
  <p class="sec-sub">Execution statistics and severity breakdown for this validation run.</p>

  <div class="kpis">
    <div class="panel kpi"><div class="label">Files Scanned</div><div class="value">{{ summary.total_files_scanned }}</div><div class="hint">{{ summary.files_fully_compliant }} fully compliant</div></div>
    <div class="panel kpi"><div class="label">Lines Evaluated</div><div class="value">{{ summary.total_lines_evaluated }}</div><div class="hint">{{ summary.non_blank_lines }} non-blank</div></div>
    <div class="panel kpi"><div class="label">Total Findings</div><div class="value">{{ summary.total_findings }}</div><div class="hint">{{ summary.unique_rules_triggered }} distinct rules</div></div>
    <div class="panel kpi with-ring">
      <div><div class="label">Line Compliance</div><div class="value">{{ summary.line_compliance_rate }}%</div><div class="hint">{{ summary.lines_with_findings }} lines flagged</div></div>
      <div class="ring" style="--p:{{ summary.line_compliance_rate }}"></div>
    </div>
    <div class="panel kpi with-ring">
      <div><div class="label">Perfect Compliance</div><div class="value">{{ summary.perfect_compliance_rate }}%</div><div class="hint">files with zero findings</div></div>
      <div class="ring" style="--p:{{ summary.perfect_compliance_rate }};--c:var(--accent)"></div>
    </div>
    <div class="panel kpi with-ring">
      <div><div class="label">Fix Coverage</div><div class="value">{{ summary.fix_coverage_rate }}%</div><div class="hint">{{ summary.fixes_suggested }} / {{ summary.total_findings }} with code fix</div></div>
      <div class="ring" style="--p:{{ summary.fix_coverage_rate }};--c:#2ecc8a"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel card-pad">
      <h3 class="card-title">Severity Breakdown</h3>
      <div class="stack" role="img" aria-label="Severity distribution">
        {% for sev, count in summary.severity_counts.items() if count %}
        <span class="{{ sev if sev in severities else 'OTHER' }}-c" style="width:{{ summary.severity_percentages[sev] }}%" title="{{ sev }}: {{ count }}"></span>
        {% endfor %}
      </div>
      <div class="sev-rows">
        {% for sev, count in summary.severity_counts.items() %}
        <div class="sev-row">
          <span><i class="dot {{ sev if sev in severities else 'OTHER' }}-c"></i>{{ sev }}</span>
          <div class="bar"><i class="{{ sev if sev in severities else 'OTHER' }}-c" style="width:{{ summary.severity_percentages[sev] }}%"></i></div>
          <span class="num">{{ count }} · {{ summary.severity_percentages[sev] }}%</span>
        </div>
        {% endfor %}
      </div>
      <div class="stats-list">
        <div>Average confidence <b>{{ '%.2f'|format(summary.average_confidence) if summary.average_confidence is not none else 'n/a' }}</b></div>
        <div>Low-confidence (&lt; {{ low_conf }}) <b>{{ summary.low_confidence_findings }}</b></div>
        <div>Lines modified <b>{{ summary.lines_modified }}</b></div>
        <div>Lines inserted <b>{{ summary.lines_inserted }}</b></div>
        <div>File-level findings <b>{{ summary.file_level_findings }}</b></div>
        <div>Fix conflicts resolved <b>{{ summary.fix_conflicts }}</b></div>
      </div>
    </div>
    <div class="panel card-pad">
      <h3 class="card-title">Findings by Rule</h3>
      {% if summary.rule_counts %}
      <table class="mini-table">
        <thead><tr><th>Rule</th><th>Severity</th><th style="text-align:right">Count</th></tr></thead>
        <tbody>
        {% for rid, info in summary.rule_counts.items() %}
          <tr><td class="rule">{{ rid }}</td><td><span class="badge {{ info.severity if info.severity in severities else 'OTHER' }}">{{ info.severity }}</span></td><td class="num">{{ info.count }}</td></tr>
        {% endfor %}
        </tbody>
      </table>
      {% else %}<div class="empty">No rules triggered.</div>{% endif %}
    </div>
  </div>
</section>

<!-- ============================ SECTION 2 ============================ -->
<section id="findings">
  <div class="sec-head"><span class="sec-num">02</span><h2>Detailed Findings &amp; Code Diffs</h2></div>
  <p class="sec-sub">Original code is shown in <span style="color:var(--del-fg)">red / strikethrough</span>; the suggested fix in <span style="color:var(--add-fg)">green</span>.</p>

  {% if findings %}
  <div class="toolbar" role="group" aria-label="Filter findings by severity">
    <button class="filter active" id="filter-all" data-sev="ALL">All ({{ findings|length }})</button>
    {% for sev, count in summary.severity_counts.items() if count %}
    <button class="filter" id="filter-{{ sev|lower }}" data-sev="{{ sev }}">{{ sev }} ({{ count }})</button>
    {% endfor %}
  </div>
  <div class="panel table-wrap">
    <table class="findings">
      <thead><tr><th>Severity</th><th>Location</th><th>Rule</th><th style="width:28%">Finding &amp; Recommendation</th><th style="width:40%">Code Diff</th><th>Confidence</th></tr></thead>
      <tbody>
      {% for f in findings %}
        <tr class="{{ f.severity if f.severity in severities else 'OTHER' }}" data-sev="{{ f.severity }}" id="finding-{{ loop.index }}">
          <td class="sev-cell"><span class="badge {{ f.severity if f.severity in severities else 'OTHER' }}">{{ f.severity }}</span></td>
          <td class="loc">L{{ f.line_number }}<small>{{ f.file_name }}</small></td>
          <td class="rule">{{ f.rule_id }}</td>
          <td>
            <p class="finding-text">{{ f.finding }}</p>
            <p class="rec"><b>Recommendation:</b> {{ f.recommendation }}</p>
          </td>
          <td>
            <div class="diff">
              <div class="diff-head">
                <span>{% if f.is_file_level %}Insert at top of file{% else %}Line {{ f.line_number }}{% endif %}</span>
                {% if f.fix_status == 'applied' %}<span class="tag applied">✓ Applied</span>
                {% elif f.fix_status == 'insertion' %}<span class="tag insertion">＋ Inserted</span>
                {% elif f.fix_status == 'superseded' %}<span class="tag superseded">Merged into {{ f.superseded_by }} fix</span>
                {% elif f.fix_status == 'unchanged' %}<span class="tag">No change</span>
                {% else %}<span class="tag none">No automated fix</span>{% endif %}
              </div>
              {% if f.old_segments %}
              <div class="diff-line {{ 'del' if f.new_segments else 'ctx' }}"><span class="g">{{ f.line_number }}</span><span class="s">{{ '−' if f.new_segments else ' ' }}</span><code>{% for text, hit in f.old_segments %}{% if hit and f.new_segments %}<mark>{{ text }}</mark>{% else %}{{ text }}{% endif %}{% endfor %}</code></div>
              {% endif %}
              {% if f.new_segments %}
              <div class="diff-line add"><span class="g">{{ f.line_number if not f.is_file_level else '' }}</span><span class="s">+</span><code>{% for text, hit in f.new_segments %}{% if hit and f.old_segments %}<mark>{{ text }}</mark>{% else %}{{ text }}{% endif %}{% endfor %}</code></div>
              {% endif %}
              {% if f.is_file_level and f.evidence %}<div class="diff-note">Evidence: {{ f.evidence }}</div>{% endif %}
            </div>
          </td>
          <td class="conf">
            {% if f.confidence is not none %}
            <div class="v">{{ '%.2f'|format(f.confidence) }}</div>
            <div class="bar"><i style="width:{{ (f.confidence * 100)|round(0) }}%;background:{{ 'var(--add-fg)' if f.confidence >= 0.9 else ('var(--medium)' if f.confidence >= low_conf else 'var(--high)') }}"></i></div>
            {% else %}<div class="v" style="color:var(--muted)">n/a</div>{% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="panel empty">✓ No compliance findings. This script passed every rule.</div>
  {% endif %}
</section>

<!-- ============================ SECTION 3 ============================ -->
<section id="updated-code">
  <div class="sec-head"><span class="sec-num">03</span><h2>Fully Updated Code</h2></div>
  <p class="sec-sub">The reconstructed script with every suggested fix applied. Highlighted lines differ from the original.</p>
  <div class="panel">
    <div class="code-head">
      <div style="display:flex;align-items:center"><div class="dots"><i></i><i></i><i></i></div><span class="code-file">{{ updated_file_name }}</span></div>
      <div class="code-actions">
        <div class="legend"><span class="lm">Modified ({{ summary.lines_modified }})</span><span class="li">Inserted ({{ summary.lines_inserted }})</span></div>
        <button class="btn" id="copy-code-btn" type="button">Copy script</button>
      </div>
    </div>
    {% if summary.fixes_suggested == 0 and findings %}
    <div class="notice">No suggested fixes were available for this run (deterministic mode or LLM fallback). Re-run with <code>--use-llm</code> to generate code fixes; the script below is unchanged.</div>
    {% endif %}
    <pre class="code"><code>{% for l in updated_lines %}<span class="cl {{ l.status }}"><span class="n">{{ loop.index }}</span><span class="o" title="Original line">{{ l.original_line_number if l.original_line_number and l.status != 'unchanged' else '' }}</span>{{ l.text }}</span>{% endfor %}</code></pre>
    <textarea id="updated-code-raw" hidden>{{ updated_code }}</textarea>
  </div>
</section>
</main>

<footer>
  <div class="warn">⚠ AI-suggested fixes must be reviewed and validated by a qualified clinical programmer before use in any regulated submission.</div>
  <div>Generated by the Clinical R Code Compliance Agent · {{ summary.generated_at }}</div>
</footer>
</div>

<script>
(function(){
  var buttons=document.querySelectorAll('.filter');
  buttons.forEach(function(b){b.addEventListener('click',function(){
    buttons.forEach(function(x){x.classList.remove('active')});b.classList.add('active');
    var sev=b.getAttribute('data-sev');
    document.querySelectorAll('table.findings tbody tr').forEach(function(r){
      r.style.display=(sev==='ALL'||r.getAttribute('data-sev')===sev)?'':'none';});
  });});
  var btn=document.getElementById('copy-code-btn');
  if(btn){btn.addEventListener('click',function(){
    var raw=document.getElementById('updated-code-raw').value;
    var done=function(){btn.textContent='Copied ✓';setTimeout(function(){btn.textContent='Copy script'},1600)};
    if(navigator.clipboard){navigator.clipboard.writeText(raw).then(done,function(){fallback(raw);done();});}
    else{fallback(raw);done();}
  });}
  function fallback(t){var ta=document.createElement('textarea');ta.value=t;document.body.appendChild(ta);ta.select();
    try{document.execCommand('copy')}catch(e){}document.body.removeChild(ta);}
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_reports(
    findings: List[FindingLike],
    file_name: str,
    output_dir: str,
    rules_dir: Union[str, Path, None] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    """
    Export findings to an Excel workbook, an HTML dashboard, and a patched R script.

    Args:
        findings: ComplianceFinding objects or deterministic analyzer dicts.
        file_name: Path to the analysed R script (read for diffs/patching).
        output_dir: Directory where reports are written.
        rules_dir: Rules directory (used to identify file-level rules).
        run_metadata: Optional display info, e.g. ``{"mode": ..., "model": ...}``.

    Returns:
        (excel_path, html_path, updated_script_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.basename(file_name))
    ext = ext or ".R"
    file_level = load_file_level_rule_ids(rules_dir)

    rows = [_normalize_finding(f, file_name) for f in findings]
    patch = apply_fixes(file_name, rows, file_level)
    summary = build_checking_summary(
        rows, [file_name], file_level, patches={os.path.basename(file_name): patch}
    )
    views = _build_finding_views(rows, patch, file_level)

    run = {"mode": "Deterministic", "model": None}
    run.update(run_metadata or {})

    # Patched script
    updated_name = f"{base_name}_updated{ext}"
    updated_path = os.path.join(output_dir, updated_name)
    with open(updated_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(patch.text)

    # Excel workbook
    excel_path = os.path.join(output_dir, f"{base_name}_compliance_report.xlsx")
    columns = [
        "file_name", "line_number", "rule_id", "severity", "finding",
        "evidence", "recommendation", "suggested_fix", "confidence",
    ]
    findings_df = pd.DataFrame(rows, columns=columns)
    summary_rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                summary_rows.append({"metric": f"{key}.{sub_key}", "value": json.dumps(sub_val) if isinstance(sub_val, dict) else sub_val})
        else:
            summary_rows.append({"metric": key, "value": value})
    code_df = pd.DataFrame(
        [
            {"line": i, "original_line": l.original_line_number, "status": l.status, "code": l.text}
            for i, l in enumerate(patch.updated_lines, start=1)
        ]
    )
    def _write_excel(path: str) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Checking Summary", index=False)
            findings_df.to_excel(writer, sheet_name="Findings", index=False)
            code_df.to_excel(writer, sheet_name="Updated Code", index=False)

    try:
        _write_excel(excel_path)
    except PermissionError:
        # Typically the workbook is open in Excel (Windows file lock).
        locked = excel_path
        excel_path = os.path.join(
            output_dir, f"{base_name}_compliance_report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        )
        print(f"Warning: {locked} is locked (open in Excel?); writing {excel_path} instead.")
        _write_excel(excel_path)

    # HTML dashboard (autoescape: R code contains <, >, & and quotes)
    html_path = os.path.join(output_dir, f"{base_name}_compliance_report.html")
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(HTML_TEMPLATE)
    html_content = template.render(
        file_name=os.path.basename(file_name),
        updated_file_name=updated_name,
        summary=summary,
        findings=views,
        updated_lines=patch.updated_lines,
        updated_code=patch.text,
        severities=SEVERITY_ORDER,
        low_conf=LOW_CONFIDENCE_THRESHOLD,
        run=run,
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    return excel_path, html_path, updated_path
