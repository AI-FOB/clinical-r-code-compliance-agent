import argparse
import asyncio
import csv
import json
import os
import sys

# Add src to the path to import the agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compliance_agent.agent import ComplianceAgent

def load_ground_truth(csv_path):
    gt = set()
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt.add((row['file_name'], int(row['line_number']), row['rule_id']))
    return gt

async def run_evaluation(scripts_dir, rules_dir, ground_truth, model, temperature, top_k, max_concurrency):
    from tqdm import tqdm
    
    agent = ComplianceAgent(
        rules_dir=rules_dir,
        model=model,
        temperature=temperature,
        top_k=top_k,
        max_concurrency=max_concurrency,
    )

    scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.R')]
    
    all_findings = []
    
    # Process files sequentially, but lines within files concurrently
    for script_name in scripts:
        script_path = os.path.join(scripts_dir, script_name)
        
        print(f"Evaluating {script_name}...")
        
        bar = None
        def on_progress(completed, total):
            nonlocal bar
            if bar is None and total:
                bar = tqdm(total=total, desc=f"Reviewing {script_name}", unit="flag", file=sys.stderr)
            if bar is not None:
                bar.n = completed
                bar.refresh()
                
        try:
            findings = await agent.aanalyze(script_path, progress_callback=on_progress)
            for f in findings:
                all_findings.append({
                    "file_name": script_name,
                    "line_number": f.line_number,
                    "rule_id": f.rule_id,
                    "finding": f.finding,
                    "evidence": f.evidence,
                    "recommendation": f.recommendation,
                    "confidence": f.confidence
                })
        finally:
            if bar is not None:
                bar.close()

    # Calculate metrics
    predicted = set((f['file_name'], f['line_number'], f['rule_id']) for f in all_findings)
    
    tp_set = predicted.intersection(ground_truth)
    fp_set = predicted - ground_truth
    fn_set = ground_truth - predicted
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print("\n--- Evaluation Results ---")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    report = {
        "metrics": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall
        },
        "false_positives": [
            {
                "file_name": f,
                "line_number": l,
                "rule_id": r,
                "details": next((item for item in all_findings if item['file_name']==f and item['line_number']==l and item['rule_id']==r), None)
            }
            for f, l, r in fp_set
        ],
        "false_negatives": [
            {"file_name": f, "line_number": l, "rule_id": r} for f, l, r in fn_set
        ]
    }
    
    report_path = os.path.join(os.path.dirname(scripts_dir), "evaluation_report.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"\nDetailed report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation framework.")
    parser.add_argument("--scripts-dir", default="evaluation/scripts", help="Directory containing R scripts.")
    parser.add_argument("--ground-truth", default="evaluation/ground_truth.csv", help="Path to ground truth CSV.")
    parser.add_argument("--rules", default="knowledge/rules", help="Directory containing rules.")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model to use.")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM temperature.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K rules.")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Max concurrency.")
    
    args = parser.parse_args()
    
    scripts_dir = os.path.abspath(args.scripts_dir)
    gt_path = os.path.abspath(args.ground_truth)
    rules_dir = os.path.abspath(args.rules)
    
    if not os.path.exists(scripts_dir) or not os.path.exists(gt_path):
        print("Error: Evaluation scripts or ground truth not found.")
        sys.exit(1)
        
    ground_truth = load_ground_truth(gt_path)
    
    asyncio.run(
        run_evaluation(
            scripts_dir, rules_dir, ground_truth, args.model, args.temperature, args.top_k, args.max_concurrency
        )
    )

if __name__ == "__main__":
    main()
