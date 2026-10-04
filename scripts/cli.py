import argparse
import asyncio
import sys
import os

# Add src to the path to import the analyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compliance_agent.tools.code_analyzer import BasicRCodeAnalyzer


def run_deterministic(filepath, rules_dir):
    analyzer = BasicRCodeAnalyzer(rules_dir)
    findings = analyzer.analyze_file(filepath)

    if not findings:
        print(f"Analyzed {filepath}: No compliance issues found.")
        return 0

    print(f"Analyzed {filepath}: Found {len(findings)} compliance issue(s):\n")
    for finding in findings:
        print(f"[{finding['severity']}] {finding['rule_id']} at line {finding['line_number']}: {finding['rule_name']}")
        print(f"  Code: {finding['line_content']}")
        print(f"  Recommendation: {finding['recommendation']}\n")
    return 1


async def run_agent_async(filepath, rules_dir, model, temperature, top_k, max_concurrency):
    # Imported lazily so the deterministic path does not load torch/ChromaDB/Ollama.
    from tqdm import tqdm
    from src.compliance_agent.agent import ComplianceAgent

    print(
        f"Running LLM agent workflow (model={model}, temperature={temperature}, "
        f"top_k={top_k}, max_concurrency={max_concurrency})...\n"
    )
    agent = ComplianceAgent(
        rules_dir=rules_dir,
        model=model,
        temperature=temperature,
        top_k=top_k,
        max_concurrency=max_concurrency,
    )

    bar = None

    def on_progress(completed, total):
        nonlocal bar
        if bar is None and total:
            bar = tqdm(total=total, desc="Reviewing flagged lines", unit="flag", file=sys.stderr)
        if bar is not None:
            bar.n = completed
            bar.refresh()

    try:
        findings = await agent.aanalyze(filepath, progress_callback=on_progress)
    finally:
        if bar is not None:
            bar.close()

    if not findings:
        print(f"Analyzed {filepath}: No compliance issues found.")
        return 0

    print(f"\nAnalyzed {filepath}: Found {len(findings)} compliance issue(s):\n")
    for f in findings:
        print(f"[{f.severity}] {f.rule_id} at line {f.line_number} (confidence {f.confidence:.2f})")
        print(f"  Finding: {f.finding}")
        print(f"  Evidence: {f.evidence}")
        print(f"  Recommendation: {f.recommendation}\n")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Analyze an R script for compliance rules.")
    parser.add_argument("filepath", help="Path to the R script to analyze.")
    parser.add_argument("--rules", default="knowledge/rules", help="Path to the directory containing JSON rules relative to project root.")
    parser.add_argument("--use-llm", action="store_true", help="Run the full agent workflow (analyzer + RAG + local Ollama LLM).")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model tag used with --use-llm (default: gemma4:e4b).")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM sampling temperature used with --use-llm (default: 1.0).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of rules retrieved per flagged snippet with --use-llm (default: 3).")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Maximum concurrent LLM requests with --use-llm (default: 4).")

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)

    rules_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.rules))
    if not os.path.exists(rules_dir):
        # Fallback if rules is absolute
        rules_dir = os.path.abspath(args.rules)

    if args.use_llm:
        exit_code = asyncio.run(
            run_agent_async(
                args.filepath, rules_dir, args.model, args.temperature, args.top_k, args.max_concurrency
            )
        )
    else:
        exit_code = run_deterministic(args.filepath, rules_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
