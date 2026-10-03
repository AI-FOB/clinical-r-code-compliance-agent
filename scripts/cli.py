import argparse
import sys
import os

# Add src to the path to import the analyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compliance_agent.tools.code_analyzer import BasicRCodeAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze an R script for compliance rules.")
    parser.add_argument("filepath", help="Path to the R script to analyze.")
    parser.add_argument("--rules", default="knowledge/rules", help="Path to the directory containing JSON rules relative to project root.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)
        
    rules_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.rules))
    if not os.path.exists(rules_dir):
        # Fallback if rules is absolute
        rules_dir = os.path.abspath(args.rules)
        
    analyzer = BasicRCodeAnalyzer(rules_dir)
    
    findings = analyzer.analyze_file(args.filepath)
    
    if not findings:
        print(f"Analyzed {args.filepath}: No compliance issues found.")
        sys.exit(0)
        
    print(f"Analyzed {args.filepath}: Found {len(findings)} compliance issue(s):\n")
    for finding in findings:
        print(f"[{finding['severity']}] {finding['rule_id']} at line {finding['line_number']}: {finding['rule_name']}")
        print(f"  Code: {finding['line_content']}")
        print(f"  Recommendation: {finding['recommendation']}\n")
        
    sys.exit(1)

if __name__ == "__main__":
    main()
