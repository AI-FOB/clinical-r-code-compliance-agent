import os
import pytest
from src.compliance_agent.tools.code_analyzer import BasicRCodeAnalyzer

@pytest.fixture
def analyzer():
    rules_dir = os.path.join(os.path.dirname(__file__), '..', 'knowledge', 'rules')
    return BasicRCodeAnalyzer(rules_dir)

def test_analyze_file_with_violations(analyzer):
    test_script = os.path.join(os.path.dirname(__file__), '..', 'examples', 'input', 'analysis_script_1.R')
    findings = analyzer.analyze_file(test_script)
    
    assert len(findings) == 9
    
    rule_ids = [f['rule_id'] for f in findings]
    assert 'COMP-001' in rule_ids
    assert 'COMP-002' in rule_ids
    assert 'COMP-003' in rule_ids
    assert 'COMP-004' in rule_ids
    assert 'COMP-005' in rule_ids
    assert 'COMP-006' in rule_ids

def test_analyze_file_clean(analyzer):
    test_script = os.path.join(os.path.dirname(__file__), '..', 'examples', 'input', 'clean_script.R')
    findings = analyzer.analyze_file(test_script)
    
    assert len(findings) == 0
