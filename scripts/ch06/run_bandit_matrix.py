#!/usr/bin/env python
"""Batch runner for Chapter 6 template bandit scenarios.

This script spins up several representative contextual-bandit settings
(`simple`, `rich`, `rich` + regularization, `rich_est`) and executes them
in parallel threads. Each scenario reuses a reproducible simulator world
seed, resolves the appropriate hyperparameters, and persists the complete
result bundle (config + metrics + captured stdout) into a JSON artifact.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from template_bandits_demo import (
    RichRegularizationConfig,
    resolve_lin_alpha,
    resolve_prior_weight,
    resolve_ts_sigma,
    run_template_bandits_experiment,
)
from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module


@dataclass
class Scenario:
    """Single experiment definition for the batch runner."""

    name: str
    description: str
    features: str
    hparam_mode: str
    n_static: int
    n_bandit: int
    world_seed: int
    bandit_base_seed: int
    rich_regularization: Optional[str] = None
    rich_blend_weight: float = 0.4
    rich_shrink: float = 0.9
    rich_quant_step: float = 0.25
    rich_clip: Tuple[float, float] = (-3.5, 3.5)
    prior_override: Optional[int] = None
    lin_alpha_override: Optional[float] = None
    ts_sigma_override: Optional[float] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "features": self.features,
            "hparam_mode": self.hparam_mode,
            "n_static": self.n_static,
            "n_bandit": self.n_bandit,
            "world_seed": self.world_seed,
            "bandit_base_seed": self.bandit_base_seed,
            "rich_regularization": self.rich_regularization,
            "rich_blend_weight": self.rich_blend_weight,
            "rich_shrink": self.rich_shrink,
            "rich_quant_step": self.rich_quant_step,
            "rich_clip": list(self.rich_clip),
            "prior_override": self.prior_override,
            "lin_alpha_override": self.lin_alpha_override,
            "ts_sigma_override": self.ts_sigma_override,
        }

    def resolve_hparams(self) -> Tuple[int, float, float]:
        prior = (
            max(0, self.prior_override)
            if self.prior_override is not None
            else resolve_prior_weight(self.hparam_mode, None)
        )
        lin_alpha = (
            max(0.1, self.lin_alpha_override)
            if self.lin_alpha_override is not None
            else resolve_lin_alpha(self.hparam_mode, None)
        )
        ts_sigma = (
            max(0.1, self.ts_sigma_override)
            if self.ts_sigma_override is not None
            else resolve_ts_sigma(self.hparam_mode, None)
        )
        return prior, lin_alpha, ts_sigma

    def build_rich_config(self) -> Optional[RichRegularizationConfig]:
        if self.features != "rich":
            return None
        if self.rich_regularization in (None, "none"):
            return None
        clip_min, clip_max = self.rich_clip
        return RichRegularizationConfig(
            mode=self.rich_regularization,
            blend_weight=self.rich_blend_weight,
            shrink=self.rich_shrink,
            quant_step=self.rich_quant_step,
            clip_min=clip_min,
            clip_max=clip_max,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch wrapper for Chapter 6 template bandit scenarios."
    )
    parser.add_argument(
        "--n-static",
        type=int,
        default=1_000,
        help="Static baseline episodes per scenario.",
    )
    parser.add_argument(
        "--n-bandit",
        type=int,
        default=20_000,
        help="Bandit training episodes per scenario.",
    )
    parser.add_argument(
        "--world-seed-base",
        type=int,
        default=2025_0701,
        help="Base simulator seed used for the first scenario (others increment).",
    )
    parser.add_argument(
        "--bandit-seed-base",
        type=int,
        default=2025_0801,
        help="Base bandit RNG seed (others offset by scenario index).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/book/drafts/ch06/data"),
        help="Directory where the aggregated JSON artifact is written.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads used for parallel execution.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel execution and run scenarios sequentially.",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="bandit_matrix",
        help="Prefix for the saved JSON filename.",
    )
    parser.add_argument(
        "--stream-logs",
        action="store_true",
        help=(
            "Stream scenario stdout to the console instead of capturing it silently. "
            "When enabled, logs are still collected for the JSON artifact."
        ),
    )
    return parser.parse_args()


def build_default_scenarios(
    n_static: int,
    n_bandit: int,
    world_seed_base: int,
    bandit_seed_base: int,
) -> List[Scenario]:
    specs = [
        {
            "name": "simple_baseline",
            "description": "Segment + query features only; pedagogical failure mode.",
            "features": "simple",
            "hparam_mode": "simple",
            "rich_regularization": None,
        },
        {
            "name": "rich_oracle_raw",
            "description": "Oracle latents with minimal smoothing (for comparison).",
            "features": "rich",
            "hparam_mode": "rich",
            "rich_regularization": "none",
        },
        {
            "name": "rich_oracle_blend",
            "description": "Oracle latents blended with segment priors + shrinkage.",
            "features": "rich",
            "hparam_mode": "rich",
            "rich_regularization": "blend",
        },
        {
            "name": "rich_oracle_quantized",
            "description": "Oracle latents quantized to mimic estimated signals.",
            "features": "rich",
            "hparam_mode": "rich_est",
            "rich_regularization": "quantized",
        },
        {
            "name": "rich_estimated",
            "description": "Estimated user latent features (production-style).",
            "features": "rich_est",
            "hparam_mode": "rich_est",
            "rich_regularization": None,
        },
    ]
    scenarios: List[Scenario] = []
    for idx, spec in enumerate(specs):
        scenarios.append(
            Scenario(
                n_static=n_static,
                n_bandit=n_bandit,
                world_seed=world_seed_base + idx,
                bandit_base_seed=bandit_seed_base + idx * 10,
                **spec,
            )
        )
    return scenarios


def _pct(delta: float, base: float) -> float:
    if base == 0.0:
        return 0.0
    return 100.0 * delta / base


def _summarize(results: Dict[str, Any]) -> Dict[str, float]:
    static_res = results["static_best"]["result"]
    lin_res = results["linucb"]["global"]
    ts_res = results["ts"]["global"]
    static_gmv = static_res["gmv"]
    summary = {
        "static_gmv": static_gmv,
        "linucb_gmv": lin_res["gmv"],
        "ts_gmv": ts_res["gmv"],
        "linucb_delta_pct": _pct(lin_res["gmv"] - static_gmv, static_gmv),
        "ts_delta_pct": _pct(ts_res["gmv"] - static_gmv, static_gmv),
    }
    return summary


class _StdoutTee(io.TextIOBase):
    """Mirrors writes to both the real stdout and a buffer."""

    def __init__(self, buffer: io.StringIO):
        super().__init__()
        self.buffer = buffer

    def write(self, s: str) -> int:
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.buffer.write(s)
        return len(s)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.buffer.flush()


def _run_scenario(scenario: Scenario, stream_logs: bool = False) -> Dict[str, Any]:
    cfg = SimulatorConfig(seed=scenario.world_seed)
    rng = np.random.default_rng(cfg.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    prior, lin_alpha, ts_sigma = scenario.resolve_hparams()
    rich_cfg = scenario.build_rich_config()

    def _execute() -> Dict[str, Any]:
        return run_template_bandits_experiment(
            cfg=cfg,
            products=products,
            n_static=scenario.n_static,
            n_bandit=scenario.n_bandit,
            feature_mode=scenario.features,
            base_seed=scenario.bandit_base_seed,
            prior_weight=prior,
            lin_alpha=lin_alpha,
            ts_sigma=ts_sigma,
            rich_regularization=rich_cfg,
        )

    buffer = io.StringIO()
    if stream_logs:
        tee = _StdoutTee(buffer)
        with redirect_stdout(tee):
            results = _execute()
    else:
        with redirect_stdout(buffer):
            results = _execute()

    metadata = scenario.metadata()
    metadata.update(
        {
            "resolved_prior_weight": prior,
            "resolved_lin_alpha": lin_alpha,
            "resolved_ts_sigma": ts_sigma,
            "resolved_rich_regularization": None
            if rich_cfg is None
            else {
                "mode": rich_cfg.mode,
                "blend_weight": rich_cfg.blend_weight,
                "shrink": rich_cfg.shrink,
                "quant_step": rich_cfg.quant_step,
                "clip": [rich_cfg.clip_min, rich_cfg.clip_max],
            },
        }
    )

    payload = {
        "scenario": metadata,
        "results": results,
        "summary": _summarize(results),
        "stdout": buffer.getvalue(),
    }
    return payload


def _print_plan(scenarios: List[Scenario]) -> None:
    print("Planned scenarios:")
    for idx, scenario in enumerate(scenarios, start=1):
        prior, lin_alpha, ts_sigma = scenario.resolve_hparams()
        reg = scenario.rich_regularization or "none"
        print(
            f"  [{idx}/{len(scenarios)}] {scenario.name:<20s} "
            f"features={scenario.features:<9s} "
            f"world_seed={scenario.world_seed:<10d} "
            f"bandit_seed={scenario.bandit_base_seed:<10d} "
            f"prior={prior:>2d} lin_alpha={lin_alpha:>4.2f} ts_sigma={ts_sigma:>4.2f} "
            f"reg={reg}"
        )


def run_scenarios(
    scenarios: List[Scenario],
    max_workers: int,
    sequential: bool,
    stream_logs: bool,
) -> List[Dict[str, Any]]:
    _print_plan(scenarios)

    if sequential or max_workers <= 1:
        outputs = []
        total = len(scenarios)
        for idx, scenario in enumerate(scenarios, start=1):
            print(f"\n[{idx}/{total}] Running scenario '{scenario.name}'...")
            outputs.append(_run_scenario(scenario, stream_logs=stream_logs))
            print(f"✓ Completed '{scenario.name}'")
        return outputs

    workers = min(max_workers, len(scenarios))
    print(
        f"\nRunning {len(scenarios)} scenarios with {workers} worker threads "
        "(parallel)..."
    )
    outputs_map: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_scenario = {
            executor.submit(_run_scenario, scenario, stream_logs): scenario
            for scenario in scenarios
        }
        completed = 0
        for future in as_completed(future_to_scenario):
            scenario = future_to_scenario[future]
            try:
                outputs_map[scenario.name] = future.result()
                completed += 1
                print(
                    f"✓ [{completed}/{len(scenarios)}] Scenario "
                    f"'{scenario.name}' completed."
                )
            except Exception as exc:  # noqa: BLE001
                print(f"✗ Scenario '{scenario.name}' failed: {exc}")
                raise
    return [outputs_map[scenario.name] for scenario in scenarios]


def main() -> None:
    args = parse_args()
    scenarios = build_default_scenarios(
        n_static=args.n_static,
        n_bandit=args.n_bandit,
        world_seed_base=args.world_seed_base,
        bandit_seed_base=args.bandit_seed_base,
    )
    results = run_scenarios(
        scenarios=scenarios,
        max_workers=args.max_workers,
        sequential=args.sequential,
        stream_logs=args.stream_logs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output_dir / f"{args.filename_prefix}_{timestamp}.json"
    )

    artifact = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_file": str(output_path),
        "parallel_workers": None if args.sequential else min(args.max_workers, len(scenarios)),
        "scenarios": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nSaved batch results to {output_path}")


if __name__ == "__main__":
    main()
