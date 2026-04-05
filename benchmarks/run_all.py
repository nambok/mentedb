#!/usr/bin/env python3
"""
MenteDB Quality Benchmark Suite

Run all benchmarks:
    python benchmarks/run_all.py

Run without LLM (no API key needed):
    python benchmarks/run_all.py --no-llm

Environment variables:
    OPENAI_API_KEY  — Required for attention_budget and noise_ratio tests
"""

import sys
import os
import argparse
import time

# Add benchmarks dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stale_belief import run_stale_belief_test
from delta_savings import run_delta_savings_test
from attention_budget import run_attention_budget_test
from noise_ratio import run_noise_ratio_test

def main():
    parser = argparse.ArgumentParser(description="MenteDB Quality Benchmarks")
    parser.add_argument("--no-llm", action="store_true", help="Skip tests requiring LLM API keys")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  MenteDB Quality Benchmark Suite")
    print("=" * 60)
    
    results = {}
    start = time.time()
    
    # No-LLM tests (always run)
    print("\n--- Engine Tests (no LLM required) ---\n")
    results["Stale Belief"] = run_stale_belief_test()
    results["Delta Savings"] = run_delta_savings_test()
    
    # LLM tests (optional)
    if not args.no_llm:
        print("\n--- LLM Quality Tests (requires OPENAI_API_KEY) ---\n")
        results["Attention Budget"] = run_attention_budget_test()
        results["Noise Ratio"] = run_noise_ratio_test()
    else:
        print("\n  [SKIP] LLM tests (--no-llm flag set)")
    
    elapsed = time.time() - start
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name:.<40} {status}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 60)
    
    # Exit code: fail if any test failed (not skipped)
    if any(r is False for r in results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
