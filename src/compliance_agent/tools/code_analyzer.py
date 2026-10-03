import os
import json
import re
from typing import List, Dict, Any

class BasicRCodeAnalyzer:
    """
    A basic static analyzer for R scripts that checks for compliance issues
    based on a set of JSON rules using regular expressions.
    """
    def __init__(self, rules_dir: str):
        """
        Initializes the analyzer by loading all rules from the specified directory.
        """
        self.rules_dir = rules_dir
        self.rules = self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        rules = []
        if not os.path.exists(self.rules_dir):
            print(f"Warning: Rules directory {self.rules_dir} does not exist.")
            return rules
            
        for filename in os.listdir(self.rules_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.rules_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        rule = json.load(f)
                        rules.append(rule)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding {filename}: {e}")
        return rules

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes an R script against the loaded compliance rules.
        """
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return findings

        # 1. Check file-level presence rules
        for rule in self.rules:
            if rule.get("rule_type") == "file_level_presence":
                missing_keywords = []
                for keyword in rule.get("required_keywords", []):
                    if keyword not in content:
                        missing_keywords.append(keyword)
                
                if missing_keywords:
                    findings.append({
                        "rule_id": rule.get("rule_id"),
                        "rule_name": rule.get("rule_name"),
                        "severity": rule.get("severity"),
                        "line_number": 1, # Default to line 1 for file-level
                        "line_content": f"Missing keywords: {', '.join(missing_keywords)}",
                        "recommendation": rule.get("recommendation")
                    })

        # 2. Check pattern-based line-by-line rules
        for line_num, line in enumerate(lines, start=1):
            for rule in self.rules:
                if rule.get("rule_type") == "file_level_presence":
                    continue
                    
                patterns = rule.get("patterns", [])
                for pattern in patterns:
                    if re.search(pattern, line):
                        findings.append({
                            "rule_id": rule.get("rule_id"),
                            "rule_name": rule.get("rule_name"),
                            "severity": rule.get("severity"),
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "recommendation": rule.get("recommendation")
                        })
                        # Break inner loop if one pattern for the rule matches
                        # so we don't report the same rule multiple times per line
                        break
                        
        return findings

if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    # Assume script is run from project root
    rules_dir = os.path.join(os.getcwd(), "knowledge", "rules")
    analyzer = BasicRCodeAnalyzer(rules_dir)
    
    test_script = os.path.join(os.getcwd(), "examples", "input", "analysis_script_1.R")
    
    if os.path.exists(test_script):
        print(f"Analyzing {test_script}...\n")
        findings = analyzer.analyze_file(test_script)
        for finding in findings:
            print(f"[{finding['severity']}] {finding['rule_id']} at line {finding['line_number']}: {finding['rule_name']}")
            print(f"Code: {finding['line_content']}")
            print(f"Recommendation: {finding['recommendation']}\n")
    else:
        print(f"Test script not found at {test_script}")
