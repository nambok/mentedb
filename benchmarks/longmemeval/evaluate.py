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

# Evaluation prompts adapted from the official LongMemEval evaluate_qa.py
JUDGE_PROMPTS = {
    "default": """You are evaluating a chat assistant's memory capabilities.

Question: {question}
Gold Answer: {answer}
Model Response: {hypothesis}

Is the model's response correct? The response is correct if it contains or is equivalent
to the gold answer. Minor phrasing differences are acceptable.
Answer with exactly "yes" or "no".""",

    "knowledge-update": """You are evaluating a chat assistant's memory capabilities.

Question: {question}
Gold Answer (most recent/updated): {answer}
Model Response: {hypothesis}

Is the model's response correct? The response is correct if it provides the UPDATED answer.
It is acceptable if the response mentions both old and new information, as long as the
updated answer is clearly present.
Answer with exactly "yes" or "no".""",

    "temporal-reasoning": """You are evaluating a chat assistant's memory capabilities.

Question: {question}
Gold Answer: {answer}
Model Response: {hypothesis}

Is the model's response correct? For temporal/counting questions, the response is correct
if the number matches the gold answer. An off-by-one difference is acceptable for
day-counting questions.
Answer with exactly "yes" or "no".""",

    "single-session-preference": """You are evaluating a chat assistant's memory capabilities.

Question: {question}
Gold Answer/Rubric: {answer}
Model Response: {hypothesis}

Is the model's response correct? The response is correct if it appropriately utilizes the
user's personal information, preferences, or context from past conversations as described
in the rubric.
Answer with exactly "yes" or "no".""",

    "abstention": """You are evaluating a chat assistant's memory capabilities.

Question: {question}
Note: This question asks about something that was NEVER discussed in conversation history.
Model Response: {hypothesis}

Is the model's response correct? The response is correct if the model identifies that it
cannot answer the question, does not have the information, or otherwise abstains from
giving a specific answer. Responses that make up an answer are INCORRECT.
Answer with exactly "yes" or "no".""",
}


def get_judge_prompt(question_data, hypothesis):
    """Select the appropriate judge prompt based on question type."""
    qtype = question_data["question_type"]
    qid = question_data["question_id"]

    if qid.endswith("_abs"):
        template = JUDGE_PROMPTS["abstention"]
    elif qtype in JUDGE_PROMPTS:
        template = JUDGE_PROMPTS[qtype]
    else:
        template = JUDGE_PROMPTS["default"]

    return template.format(
        question=question_data["question"],
        answer=question_data.get("answer", ""),
        hypothesis=hypothesis,
    )


def judge_answer(client, provider, question_data, hypothesis):
    """Use an LLM to judge whether the hypothesis is correct."""
    prompt = get_judge_prompt(question_data, hypothesis)

    try:
        if provider == "openai":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            verdict = response.choices[0].message.content.strip().lower()
        elif provider == "anthropic":
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            verdict = response.content[0].text.strip().lower()
        else:
            return 0

        return 1 if verdict.startswith("yes") else 0
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
    # Support both OpenAI and Anthropic as judge
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from harness import get_llm_client

    llm_client, llm_provider = get_llm_client()
    if not llm_client:
        print("Need OPENAI_API_KEY or ANTHROPIC_API_KEY for judge evaluation.")
        sys.exit(1)

    judge_model = "gpt-4o" if llm_provider == "openai" else "claude-sonnet-4-20250514"

    reference = load_reference(variant)
    hypotheses = load_hypotheses(hypothesis_file)

    print(f"Evaluating {len(hypotheses)} predictions against {variant} reference...")
    print(f"Judge: {judge_model}\n")

    eval_results = []
    correct = 0

    eval_log_file = hypothesis_file + f".eval-results-{judge_model}"

    from tqdm import tqdm
    for entry in tqdm(hypotheses, desc="Judging"):
        qid = entry["question_id"]
        hypothesis = entry["hypothesis"]
        ref = reference.get(qid)

        if not ref:
            print(f"  Warning: {qid} not found in reference, skipping")
            continue

        label = judge_answer(llm_client, llm_provider, ref, hypothesis)
        correct += label

        eval_entry = {
            "question_id": qid,
            "hypothesis": hypothesis,
            "autoeval_label": {"model": judge_model, "label": label},
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
        f.write(f"Questions: {len(eval_results)}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.1f}%\n")

    print(f"\nEval log: {eval_log_file}")
    print(f"Metrics: {metrics_file}")

    return metrics


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
