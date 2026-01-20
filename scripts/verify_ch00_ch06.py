#!/usr/bin/env python3
"""Run Chapters 0-6 code sequentially and save logs/artifacts.

This script creates a dedicated output folder per run and captures:
- stdout/stderr logs for every step
- pytest XML reports
- generated artifacts (plots + JSON summaries) in a separate tree
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    step_id: str
    name: str
    cmd: List[str]
    timeout_s: int


def _run_command(
    step: Step,
    index: int,
    total: int,
    run_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{index:02d}_{step.step_id}.log"

    header = [
        f"STEP {index}/{total}: {step.name}",
        f"CMD: {' '.join(step.cmd)}",
        f"CWD: {PROJECT_ROOT}",
        f"START: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
    ]

    print(f"[{index}/{total}] {step.name}")
    start = time.time()
    timed_out = False
    returncode = None
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(header))
        log_file.flush()
        try:
            result = subprocess.run(
                step.cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=step.timeout_s,
                check=False,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            log_file.write("\n[ERROR] TimeoutExpired\n")
        except Exception as exc:  # noqa: BLE001
            returncode = 1
            log_file.write(f"\n[ERROR] {exc}\n")

    duration_s = time.time() - start
    return {
        "step_id": step.step_id,
        "name": step.name,
        "cmd": step.cmd,
        "log": str(log_path),
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": round(duration_s, 2),
    }


def _safe_git_rev() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _build_steps(
    *,
    mode: str,
    artifacts_dir: Path,
    pytest_marker: str | None,
    ch06_n_static: int,
    ch06_n_bandit: int,
) -> List[Step]:
    python = sys.executable
    steps: List[Step] = []

    def add(step_id: str, name: str, cmd: List[str], timeout_s: int) -> None:
        steps.append(Step(step_id=step_id, name=name, cmd=cmd, timeout_s=timeout_s))

    def pytest_step(step_id: str, name: str, test_path: str) -> None:
        junit_path = artifacts_dir / "pytest" / f"{step_id}.xml"
        basetemp = artifacts_dir / "pytest" / "tmp" / step_id
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        basetemp.mkdir(parents=True, exist_ok=True)
        cmd = [
            python,
            "-m",
            "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--junitxml",
            str(junit_path),
            "--basetemp",
            str(basetemp),
        ]
        if pytest_marker:
            cmd.extend(["-m", pytest_marker])
        add(step_id, name, cmd, timeout_s=600)

    # Tests (Chapter 0-6)
    pytest_step("pytest_ch00", "Pytest: Chapter 0", "tests/ch00")
    pytest_step("pytest_ch01", "Pytest: Chapter 1", "tests/ch01")
    pytest_step("pytest_ch02", "Pytest: Chapter 2", "tests/ch02")
    pytest_step("pytest_ch03", "Pytest: Chapter 3", "tests/ch03")
    pytest_step("pytest_ch05", "Pytest: Chapter 5", "tests/ch05")
    pytest_step("pytest_ch06", "Pytest: Chapter 6", "tests/ch06")

    # Chapter 0
    ch00_plot = artifacts_dir / "ch00" / "learning_curves.png"
    ch00_plot.parent.mkdir(parents=True, exist_ok=True)
    add(
        "ch00_toy_problem",
        "Ch00: Toy Problem (book figure)",
        [
            python,
            "scripts/ch00/toy_problem_solution.py",
            "--chapter0",
            "--seed",
            "42",
            "--plot-path",
            str(ch00_plot),
        ],
        timeout_s=300,
    )

    if mode == "full":
        add(
            "ch00_labs_all",
            "Ch00: Lab Solutions (all)",
            [python, "scripts/ch00/lab_solutions.py", "--all"],
            timeout_s=600,
        )
    else:
        add(
            "ch00_lab_0_1",
            "Ch00: Lab 0.1 (tabular boost search)",
            [python, "scripts/ch00/lab_solutions.py", "--exercise", "lab0.1"],
            timeout_s=300,
        )

    # Chapter 1
    if mode == "full":
        add(
            "ch01_labs_all",
            "Ch01: Lab Solutions (all)",
            [python, "scripts/ch01/lab_solutions.py", "--all"],
            timeout_s=600,
        )
    else:
        add(
            "ch01_lab_1_1",
            "Ch01: Lab 1.1",
            [python, "scripts/ch01/lab_solutions.py", "--lab", "1.1"],
            timeout_s=300,
        )
        add(
            "ch01_lab_1_2",
            "Ch01: Lab 1.2",
            [python, "scripts/ch01/lab_solutions.py", "--lab", "1.2"],
            timeout_s=300,
        )

    # Chapter 2
    if mode == "full":
        add(
            "ch02_labs_all",
            "Ch02: Lab Solutions (all)",
            [python, "scripts/ch02/lab_solutions.py", "--all"],
            timeout_s=900,
        )
    else:
        add(
            "ch02_lab_2_1",
            "Ch02: Lab 2.1",
            [python, "scripts/ch02/lab_solutions.py", "--lab", "2.1"],
            timeout_s=300,
        )
        add(
            "ch02_lab_2_2",
            "Ch02: Lab 2.2",
            [python, "scripts/ch02/lab_solutions.py", "--lab", "2.2"],
            timeout_s=300,
        )

    # Chapter 3
    if mode == "full":
        add(
            "ch03_labs_all",
            "Ch03: Lab Solutions (all)",
            [python, "scripts/ch03/lab_solutions.py", "--all"],
            timeout_s=900,
        )
    else:
        add(
            "ch03_lab_3_1",
            "Ch03: Lab 3.1",
            [python, "scripts/ch03/lab_solutions.py", "--lab", "3.1"],
            timeout_s=300,
        )
        add(
            "ch03_lab_3_2",
            "Ch03: Lab 3.2",
            [python, "scripts/ch03/lab_solutions.py", "--lab", "3.2"],
            timeout_s=300,
        )

    # Chapter 4
    add(
        "ch04_demo",
        "Ch04: Generative World Demo",
        [python, "scripts/ch04/ch04_demo.py"],
        timeout_s=300,
    )

    # Chapter 5
    add(
        "ch05_demo",
        "Ch05: Relevance/Features/Reward Demo",
        [python, "scripts/ch05/ch05_demo.py"],
        timeout_s=300,
    )
    add(
        "ch05_validate",
        "Ch05: Validation Script",
        [python, "scripts/validate_ch05.py"],
        timeout_s=300,
    )

    # Chapter 6
    add(
        "ch06_template_demo_simple",
        "Ch06: Template Bandits Demo (simple features)",
        [
            python,
            "scripts/ch06/template_bandits_demo.py",
            "--n-static",
            str(ch06_n_static),
            "--n-bandit",
            str(ch06_n_bandit),
            "--features",
            "simple",
        ],
        timeout_s=1200,
    )

    if mode == "full":
        add(
            "ch06_template_demo_rich",
            "Ch06: Template Bandits Demo (rich features)",
            [
                python,
                "scripts/ch06/template_bandits_demo.py",
                "--n-static",
                str(ch06_n_static),
                "--n-bandit",
                str(ch06_n_bandit),
                "--features",
                "rich",
            ],
            timeout_s=1200,
        )
        add(
            "ch06_template_demo_rich_est",
            "Ch06: Template Bandits Demo (rich_est features)",
            [
                python,
                "scripts/ch06/template_bandits_demo.py",
                "--n-static",
                str(ch06_n_static),
                "--n-bandit",
                str(ch06_n_bandit),
                "--features",
                "rich_est",
            ],
            timeout_s=1200,
        )

    ch06_data_dir = artifacts_dir / "ch06" / "data"
    ch06_data_dir.mkdir(parents=True, exist_ok=True)
    add(
        "ch06_bandit_matrix",
        "Ch06: Bandit Matrix (JSON artifact)",
        [
            python,
            "scripts/ch06/run_bandit_matrix.py",
            "--n-static",
            str(ch06_n_static),
            "--n-bandit",
            str(ch06_n_bandit),
            "--output-dir",
            str(ch06_data_dir),
            "--sequential",
        ],
        timeout_s=1800,
    )
    add(
        "ch06_compute_arc",
        "Ch06: Compute Arc (JSON + figures)",
        [
            python,
            "scripts/ch06/ch06_compute_arc.py",
            "--n-static",
            str(ch06_n_static),
            "--n-bandit",
            str(ch06_n_bandit),
            "--out-dir",
            str(ch06_data_dir),
        ],
        timeout_s=1800,
    )

    if mode == "full":
        add(
            "ch06_labs_all",
            "Ch06: Lab Solutions (all)",
            [python, "-m", "scripts.ch06.lab_solutions", "--all"],
            timeout_s=2400,
        )
        add(
            "ch06_ex_6_6b",
            "Ch06: Exercise 6.6b (diversity helpful case)",
            [python, "-m", "scripts.ch06.lab_solutions", "--exercise", "6.6b"],
            timeout_s=1200,
        )

    return steps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Chapters 0-6 code sequentially and save logs/artifacts."
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="full",
        help="Run scope: fast (smoke tests) or full (all labs + demos).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("verification_runs/ch00_06"),
        help="Root directory for saved outputs (a timestamped subfolder is created).",
    )
    parser.add_argument(
        "--ch06-n-static",
        type=int,
        default=None,
        help="Override n_static for Chapter 6 runs.",
    )
    parser.add_argument(
        "--ch06-n-bandit",
        type=int,
        default=None,
        help="Override n_bandit for Chapter 6 runs.",
    )
    args = parser.parse_args()

    defaults = {
        "fast": {"n_static": 200, "n_bandit": 1000, "pytest_marker": "not slow"},
        "full": {"n_static": 2000, "n_bandit": 20000, "pytest_marker": None},
    }
    mode_cfg = defaults[args.mode]
    ch06_n_static = args.ch06_n_static or mode_cfg["n_static"]
    ch06_n_bandit = args.ch06_n_bandit or mode_cfg["n_bandit"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / f"run_{timestamp}"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("XDG_CACHE_HOME", str(run_dir / "cache"))
    env.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib"))

    steps = _build_steps(
        mode=args.mode,
        artifacts_dir=artifacts_dir,
        pytest_marker=mode_cfg["pytest_marker"],
        ch06_n_static=ch06_n_static,
        ch06_n_bandit=ch06_n_bandit,
    )

    metadata = {
        "mode": args.mode,
        "run_dir": str(run_dir),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": _safe_git_rev(),
        "ch06_n_static": ch06_n_static,
        "ch06_n_bandit": ch06_n_bandit,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    results: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        results.append(_run_command(step, idx, len(steps), run_dir, env))

    summary = {
        "metadata": metadata,
        "results": results,
        "ok": all(r["returncode"] == 0 for r in results),
    }
    summary_path = run_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. Summary written to {summary_path}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
