#!/usr/bin/env python
"""
Full codebase validation script.

Run with: python scripts/validate_all.py [--quick | --full]

--quick: Fast unit tests only (~10 seconds)
--full:  All tests + key scripts (~5-10 minutes)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_cmd(cmd: str, description: str, timeout: int = 300) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"[RUNNING] {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            capture_output=False,
        )
        success = result.returncode == 0
        status = "PASSED" if success else "FAILED"
        print(f"[{status}] {description}")
        return success
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description}")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False


def quick_validation() -> bool:
    """Run fast unit tests only."""
    print("\n" + "=" * 70)
    print("QUICK VALIDATION (~10 seconds)")
    print("=" * 70)

    # Fast tests (skip slow integration tests)
    fast_tests = [
        "tests/ch00",
        "tests/ch01",
        "tests/ch02",
        "tests/ch03",
        "tests/ch05",
        "tests/ch07",
        "tests/ch08",
        "tests/ch09",
        "tests/ch11",
        "tests/ch06/test_linucb.py",
        "tests/ch06/test_templates.py",
        "tests/ch06/test_thompson_sampling.py",
        "tests/ch06/test_integration.py",  # Ch6 integration (fast, ~5s)
        "tests/test_catalog_stats.py",
        "tests/test_env_basic.py",
    ]

    cmd = f"pytest {' '.join(fast_tests)} -v --tb=short"
    return run_cmd(cmd, "Fast unit tests", timeout=60)


def medium_validation() -> bool:
    """Run unit tests + key scripts."""
    results = []

    # Unit tests
    cmd = (
        "pytest tests/ "
        "--ignore=tests/ch06/test_feature_modes_integration.py "
        "--ignore=tests/ch06/test_integration.py "
        "-v --tb=short"
    )
    results.append(run_cmd(cmd, "Unit tests (excluding slow integration)", timeout=120))

    # Key scripts
    scripts = [
        ("python scripts/ch01/lab_solutions.py --lab 1.1", "Ch01 Lab 1.1"),
        ("python scripts/ch03/lab_solutions.py --all", "Ch03 Labs"),
    ]

    for cmd, desc in scripts:
        results.append(run_cmd(cmd, desc, timeout=60))

    return all(results)


def full_validation() -> bool:
    """Run all tests and key scripts."""
    results = []

    print("\n" + "=" * 70)
    print("FULL VALIDATION (~5-10 minutes)")
    print("=" * 70)

    # All unit tests (skip only very slow integration)
    cmd = (
        "pytest tests/ "
        "--ignore=tests/ch06/test_feature_modes_integration.py "
        "-v --tb=short"
    )
    results.append(run_cmd(cmd, "All unit tests", timeout=300))

    # Chapter scripts
    scripts = [
        ("python scripts/ch00/toy_problem_solution.py", "Ch00 Q-Learning"),
        ("python scripts/ch01/lab_solutions.py --all", "Ch01 Labs"),
        ("python scripts/ch02/lab_solutions.py --all", "Ch02 Labs"),
        ("python scripts/ch03/lab_solutions.py --all", "Ch03 Labs"),
        ("python scripts/ch04/ch04_demo.py", "Ch04 Demo"),
        ("python scripts/ch05/ch05_demo.py", "Ch05 Demo"),
    ]

    for cmd, desc in scripts:
        results.append(run_cmd(cmd, desc, timeout=120))

    return all(results)


def ch01_verification() -> bool:
    """Run Chapter 1 specific verification."""
    print("\n" + "=" * 70)
    print("CHAPTER 1 SPECIFIC VERIFICATION")
    print("=" * 70)

    results = []

    # Tests
    results.append(run_cmd(
        "pytest tests/ch01/test_reward_examples.py -v",
        "Ch01 Tests",
        timeout=30
    ))

    # Labs
    results.append(run_cmd(
        "python scripts/ch01/lab_solutions.py --all",
        "Ch01 All Labs",
        timeout=60
    ))

    # Verification greps
    verifications = [
        ('grep -c "Two strategic quantities" docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md || echo "0"',
         "STRAT clarification note exists"),
        ('grep -rn "strat_exposure" docs/book/ch01/ tests/ch01/ scripts/ch01/ 2>/dev/null | grep -vc "plan\\|update" || echo "0"',
         "No strat_exposure remnants (expect 0)"),
    ]

    for cmd, desc in verifications:
        run_cmd(cmd, desc, timeout=10)

    return all(results)


def ch06_verification() -> bool:
    """Run Chapter 6 specific verification (tiered)."""
    print("\n" + "=" * 70)
    print("CHAPTER 6 SPECIFIC VERIFICATION")
    print("=" * 70)

    results = []

    # Tier 1+2: Fast unit + integration tests (~1s)
    results.append(run_cmd(
        "pytest tests/ch06/test_linucb.py tests/ch06/test_templates.py "
        "tests/ch06/test_thompson_sampling.py tests/ch06/test_integration.py -v",
        "Ch06 Tier 1+2: Unit + Integration (27 tests)",
        timeout=30
    ))

    # Tier 3: Narrative smoke test (~60s)
    results.append(run_cmd(
        "python scripts/ch06/template_bandits_demo.py "
        "--n-static 200 --n-bandit 1000 --features simple",
        "Ch06 Tier 3: Narrative smoke (simple features FAIL)",
        timeout=180
    ))

    # Template library sanity check (simple assertion via test)
    results.append(run_cmd(
        "pytest tests/ch06/test_templates.py::test_create_standard_templates -v",
        "Ch06 Template library (8 templates)",
        timeout=30
    ))

    return all(results)


def main():
    parser = argparse.ArgumentParser(description="Validate codebase")
    parser.add_argument("--quick", action="store_true", help="Fast tests only (~10s)")
    parser.add_argument("--medium", action="store_true", help="Tests + key scripts (~2min)")
    parser.add_argument("--full", action="store_true", help="All tests + scripts (~5-10min)")
    parser.add_argument("--ch01", action="store_true", help="Chapter 1 specific checks")
    parser.add_argument("--ch06", action="store_true", help="Chapter 6 specific checks (~2min)")

    args = parser.parse_args()

    # Default to quick if no args
    if not any([args.quick, args.medium, args.full, args.ch01, args.ch06]):
        args.quick = True

    success = True

    if args.quick:
        success = quick_validation()
    elif args.medium:
        success = medium_validation()
    elif args.full:
        success = full_validation()
    elif args.ch01:
        success = ch01_verification()
    elif args.ch06:
        success = ch06_verification()

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - Check output above")
    print("=" * 70)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
