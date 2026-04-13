#!/usr/bin/env python3
"""
Combine and evaluate all paginated LongMemEval benchmark results.

Finds all hypotheses_q*.jsonl files, combines them, checks for gaps,
and runs the full evaluation with a combined report.

Usage:
    python benchmarks/longmemeval/evaluate_all.py [--dataset s]
"""

import argparse
import glob
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def load_dataset(variant):
    filepath = os.path.join(SCRIPT_DIR, "data", DATASET_FILES[variant])
    with open(filepath) as f:
        return json.load(f)


def find_page_files():
    """Find all paginated hypothesis files and return sorted by offset."""
    pattern = os.path.join(RESULTS_DIR, "hypotheses_q*.jsonl")
    files = glob.glob(pattern)
    # Filter out eval results and reports
    files = [f for f in files if not f.endswith("_report.md")
             and "eval-results" not in f]

    parsed = []
    for f in files:
        match = re.search(r"hypotheses_q(\d+)-(\d+)\.jsonl$", f)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            parsed.append((start, end, f))

    return sorted(parsed, key=lambda x: x[0])


def combine_results(page_files):
    """Combine all page files into one list, checking for duplicates."""
    all_results = []
    seen_ids = set()

    for start, end, filepath in page_files:
        count = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                qid = entry["question_id"]
                if qid not in seen_ids:
                    seen_ids.add(qid)
                    all_results.append(entry)
                    count += 1
        print(f"  ✓ q{start}-{end}: {count} questions ({filepath})")

    return all_results, seen_ids


def check_coverage(seen_ids, dataset):
    """Check which questions are missing."""
    all_ids = {q["question_id"] for q in dataset}
    missing = all_ids - seen_ids
    extra = seen_ids - all_ids

    if missing:
        print(f"\n  ⚠️  Missing {len(missing)} questions:")
        # Group by ranges to show which pages need to run
        missing_indices = []
        for i, q in enumerate(dataset):
            if q["question_id"] in missing:
                missing_indices.append(i)
        if missing_indices:
            # Find contiguous ranges
            ranges = []
            start = missing_indices[0]
            prev = start
            for idx in missing_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    ranges.append((start, prev + 1))
                    start = idx
                    prev = idx
            ranges.append((start, prev + 1))

            print("  Run these to fill gaps:")
            for s, e in ranges:
                # Round to page boundaries
                page_start = (s // 50) * 50
                page_end = min(((e - 1) // 50 + 1) * 50, len(dataset))
                print(f"    .venv/bin/python benchmarks/longmemeval/run_benchmark.py "
                      f"--offset {page_start} --limit {page_end - page_start} --workers 3")

    if extra:
        print(f"\n  ⚠️  {len(extra)} questions not in reference (ignored)")

    return len(missing)


def main():
    parser = argparse.ArgumentParser(
        description="Combine and evaluate all paginated LongMemEval results")
    parser.add_argument("--dataset", default="s", choices=["s", "m", "oracle"],
                        help="Reference dataset variant (default: s)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip combining, just evaluate the combined file")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    total = len(dataset)

    print(f"{'='*60}")
    print(f"  LongMemEval — Combine & Evaluate All Pages")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset} ({total} questions)")
    print()

    # Find and combine page files
    page_files = find_page_files()
    if not page_files:
        print("  No page files found in results/")
        print("  Expected files like: hypotheses_q0-50.jsonl, hypotheses_q50-100.jsonl, ...")
        print(f"\n  Run: .venv/bin/python benchmarks/longmemeval/run_benchmark.py "
              f"--offset 0 --limit 50 --workers 3")
        sys.exit(1)

    print(f"  Found {len(page_files)} page files:\n")
    all_results, seen_ids = combine_results(page_files)

    print(f"\n  Total unique questions: {len(all_results)} / {total}")

    # Check coverage
    missing = check_coverage(seen_ids, dataset)

    # Write combined file
    combined_file = os.path.join(RESULTS_DIR, "hypotheses_combined.jsonl")
    with open(combined_file, "w") as f:
        for entry in all_results:
            f.write(json.dumps(entry) + "\n")
    print(f"\n  Combined file: {combined_file}")

    if missing > 0:
        pct = (len(all_results) / total) * 100
        print(f"\n  Coverage: {pct:.0f}% ({len(all_results)}/{total})")
        print(f"  Evaluating what we have ({len(all_results)} questions)...")
    else:
        print(f"\n  ✅ Full coverage! All {total} questions answered.")

    # Run evaluation
    print(f"\n{'='*60}")
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate import run_evaluation
    run_evaluation(combined_file, variant=args.dataset)


if __name__ == "__main__":
    main()
