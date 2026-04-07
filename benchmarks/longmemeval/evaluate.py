#!/usr/bin/env python3
"""
Evaluate LongMemEval results using the official GPT-4o judge.

This wraps the official LongMemEval evaluation methodology:
each hypothesis is judged by GPT-4o against the gold answer.

Usage:
    python benchmarks/longmemeval/evaluate.py results/hypotheses.jsonl [--dataset s]

Requires:
    OPENAI_API_KEY (GPT-4o is used as the judge, matching the official eval)
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

# Evaluation prompts: EXACT copies from the official LongMemEval evaluate_qa.py
# Source: https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
JUDGE_PROMPTS = {
    "single-session-user": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only.",

    "single-session-assistant": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only.",

    "multi-session": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only.",

    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only.",

    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only.",

    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only.",

    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {hypothesis}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}

# Official judge model — must match for comparable results
OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"


def get_judge_prompt(question_data, hypothesis):
    """Select the appropriate judge prompt based on question type.

    Uses the EXACT official prompts from LongMemEval evaluate_qa.py.
    """
    qtype = question_data["question_type"]
    qid = question_data["question_id"]

    if qid.endswith("_abs"):
        template = JUDGE_PROMPTS["abstention"]
    elif qtype in JUDGE_PROMPTS:
        template = JUDGE_PROMPTS[qtype]
    else:
        # Fallback for any unknown type — use the same generic prompt as
        # single-session-user (the official code would raise NotImplementedError)
        template = JUDGE_PROMPTS["single-session-user"]

    return template.format(
        question=question_data["question"],
        answer=question_data.get("answer", ""),
        hypothesis=hypothesis,
    )


def judge_answer(client, question_data, hypothesis):
    """Use GPT-4o to judge whether the hypothesis is correct.

    Always uses gpt-4o-2024-08-06 to match the official LongMemEval evaluation.
    Requires OPENAI_API_KEY — this is non-negotiable for comparable results.
    """
    prompt = get_judge_prompt(question_data, hypothesis)

    try:
        response = client.chat.completions.create(
            model=OFFICIAL_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            n=1,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return 1 if "yes" in verdict else 0
    except Exception as e:
        print(f"  Judge error on {question_data['question_id']}: {e}")
        return 0


def load_reference(variant):
    filepath = os.path.join(DATA_DIR, DATASET_FILES[variant])
    if not os.path.exists(filepath):
        print(f"Reference dataset not found: {filepath}")
        print("Run: python benchmarks/longmemeval/download_data.py")
        sys.exit(1)
    with open(filepath) as f:
        data = json.load(f)
    return {q["question_id"]: q for q in data}


def load_hypotheses(filepath):
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_metrics(eval_results, reference):
    """Compute per-type and overall accuracy."""
    type_correct = {}
    type_total = {}
    abs_correct = 0
    abs_total = 0

    for entry in eval_results:
        qid = entry["question_id"]
        label = entry.get("label", 0)
        ref = reference.get(qid, {})
        qtype = ref.get("question_type", "unknown")

        if qid.endswith("_abs"):
            abs_total += 1
            abs_correct += label
        else:
            type_correct[qtype] = type_correct.get(qtype, 0) + label
            type_total[qtype] = type_total.get(qtype, 0) + 1

    metrics = {}

    print("\nResults by Category")
    print("=" * 60)

    type_accuracies = []
    for qtype in sorted(type_total.keys()):
        acc = type_correct.get(qtype, 0) / type_total[qtype] * 100
        type_accuracies.append(acc)
        metrics[qtype] = acc
        print(f"  {qtype:.<40} {acc:5.1f}% ({type_correct.get(qtype, 0)}/{type_total[qtype]})")

    if abs_total > 0:
        abs_acc = abs_correct / abs_total * 100
        metrics["abstention"] = abs_acc
        print(f"  {'abstention':.<40} {abs_acc:5.1f}% ({abs_correct}/{abs_total})")
        type_accuracies.append(abs_acc)

    total_correct = sum(type_correct.values()) + abs_correct
    total_questions = sum(type_total.values()) + abs_total
    overall = total_correct / total_questions * 100 if total_questions > 0 else 0
    task_avg = sum(type_accuracies) / len(type_accuracies) if type_accuracies else 0

    metrics["overall"] = overall
    metrics["task_averaged"] = task_avg

    print(f"\n  {'Task-Averaged Accuracy':.<40} {task_avg:5.1f}%")
    print(f"  {'Overall Accuracy':.<40} {overall:5.1f}% ({total_correct}/{total_questions})")
    print("=" * 60)

    return metrics


def run_evaluation(hypothesis_file, variant="s"):
    # Official evaluation requires OpenAI GPT-4o as judge
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY is required for evaluation.")
        print("The official LongMemEval eval uses gpt-4o-2024-08-06 as judge.")
        print("Using a different judge model would make results non-comparable.")
        sys.exit(1)

    try:
        import openai
        judge_client = openai.OpenAI()
    except ImportError:
        print("openai package required for evaluation: pip install openai")
        sys.exit(1)

    reference = load_reference(variant)
    hypotheses = load_hypotheses(hypothesis_file)

    print(f"Evaluating {len(hypotheses)} predictions against {variant} reference...")
    print(f"Judge: {OFFICIAL_JUDGE_MODEL} (official LongMemEval judge)\n")

    eval_results = []
    correct = 0

    eval_log_file = hypothesis_file + f".eval-results-{OFFICIAL_JUDGE_MODEL}"

    from tqdm import tqdm
    for entry in tqdm(hypotheses, desc="Judging"):
        qid = entry["question_id"]
        hypothesis = entry["hypothesis"]
        ref = reference.get(qid)

        if not ref:
            print(f"  Warning: {qid} not found in reference, skipping")
            continue

        label = judge_answer(judge_client, ref, hypothesis)
        correct += label

        eval_entry = {
            "question_id": qid,
            "hypothesis": hypothesis,
            "autoeval_label": {"model": OFFICIAL_JUDGE_MODEL, "label": label},
            "label": label,
        }
        eval_results.append(eval_entry)

        # Write incrementally
        with open(eval_log_file, "a") as f:
            f.write(json.dumps(eval_entry) + "\n")

    # Write clean final file
    with open(eval_log_file, "w") as f:
        for entry in eval_results:
            f.write(json.dumps(entry) + "\n")

    metrics = compute_metrics(eval_results, reference)

    # Save metrics summary
    metrics_file = os.path.join(os.path.dirname(hypothesis_file), "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("LongMemEval Results (MenteDB)\n")
        f.write(f"Dataset: {variant}\n")
        f.write(f"Judge: {OFFICIAL_JUDGE_MODEL}\n")
        f.write(f"Questions: {len(eval_results)}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.1f}%\n")

    # Generate shareable markdown report
    report_file = hypothesis_file.replace(".jsonl", "") + "_report.md"
    generate_report(report_file, eval_results, metrics, reference, variant)

    print(f"\nEval log: {eval_log_file}")
    print(f"Metrics: {metrics_file}")
    print(f"Report:  {report_file}")

    return metrics


def generate_report(report_file, eval_results, metrics, reference, variant):
    """Generate a shareable markdown report with full results."""
    from datetime import datetime

    total_correct = sum(1 for e in eval_results if e.get("label") == 1)
    total = len(eval_results)
    overall = total_correct / total * 100 if total else 0

    with open(report_file, "w") as f:
        f.write("# LongMemEval Benchmark Report — MenteDB\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Dataset:** LongMemEval-{variant.upper()} ({total} questions)\n")
        f.write(f"**Judge:** {OFFICIAL_JUDGE_MODEL} (official)\n\n")

        f.write("## Configuration\n\n")
        f.write("| Component | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Extraction model | {os.environ.get('MENTEDB_LLM_MODEL', 'default')} |\n")
        f.write(f"| Extraction provider | {os.environ.get('MENTEDB_LLM_PROVIDER', 'default')} |\n")
        f.write(f"| Embedding | {os.environ.get('EMBEDDING_PROVIDER', 'openai')} / {os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')} |\n")
        f.write("| BM25 hybrid search | ON |\n")
        f.write("| Fact-augmented keys | ON |\n")
        f.write("| Turn decomposition | ON |\n")
        f.write("| Multi-query RRF | ON |\n")
        f.write("| Time-aware filter | ON |\n\n")

        f.write("## Results\n\n")
        f.write(f"**Overall Accuracy: {overall:.1f}% ({total_correct}/{total})**\n\n")
        f.write("| Category | Accuracy | Correct | Total |\n")
        f.write("|---|---|---|---|\n")

        type_results = {}
        for entry in eval_results:
            qid = entry["question_id"]
            ref = reference.get(qid, {})
            qtype = ref.get("question_type", "unknown")
            if qtype not in type_results:
                type_results[qtype] = {"correct": 0, "total": 0}
            type_results[qtype]["total"] += 1
            if entry.get("label") == 1:
                type_results[qtype]["correct"] += 1

        for qtype in sorted(type_results.keys()):
            r = type_results[qtype]
            acc = r["correct"] / r["total"] * 100 if r["total"] else 0
            f.write(f"| {qtype} | {acc:.1f}% | {r['correct']} | {r['total']} |\n")

        task_avg = metrics.get("task_averaged", 0)
        f.write(f"\n**Task-Averaged Accuracy: {task_avg:.1f}%**\n\n")

        # Per-question details
        f.write("## Per-Question Details\n\n")
        f.write("| # | QID | Category | Result | Question (truncated) |\n")
        f.write("|---|---|---|---|---|\n")
        for i, entry in enumerate(eval_results, 1):
            qid = entry["question_id"]
            ref = reference.get(qid, {})
            qtype = ref.get("question_type", "unknown")
            result = "✅" if entry.get("label") == 1 else "❌"
            question = ref.get("question", "")[:60].replace("|", "\\|")
            f.write(f"| {i} | {qid[:8]} | {qtype} | {result} | {question} |\n")

        # Show failures in detail
        failures = [e for e in eval_results if e.get("label") != 1]
        if failures:
            f.write("\n## Failures\n\n")
            for entry in failures:
                qid = entry["question_id"]
                ref = reference.get(qid, {})
                f.write(f"### {qid}\n\n")
                f.write(f"**Question:** {ref.get('question', 'N/A')}\n\n")
                f.write(f"**Gold answer:** {ref.get('answer', 'N/A')}\n\n")
                f.write(f"**Our answer:** {entry.get('hypothesis', 'N/A')}\n\n")

        f.write("\n---\n")
        f.write(f"*Generated by MenteDB LongMemEval benchmark. ")
        f.write(f"Judge: {OFFICIAL_JUDGE_MODEL} (official LongMemEval methodology).*\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval results")
    parser.add_argument("hypothesis_file", help="Path to hypotheses.jsonl")
    parser.add_argument("--dataset", default="s", choices=["s", "m", "oracle"],
                        help="Reference dataset variant (default: s)")
    args = parser.parse_args()

    if not os.path.exists(args.hypothesis_file):
        print(f"File not found: {args.hypothesis_file}")
        sys.exit(1)

    run_evaluation(args.hypothesis_file, variant=args.dataset)


if __name__ == "__main__":
    main()
