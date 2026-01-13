#!/usr/bin/env python
"""GPU-accelerated batch runner for Chapter 6 template bandit scenarios."""

from __future__ import annotations

import argparse
import json
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from template_bandits_gpu import (
    RichRegularizationConfig,
    resolve_lin_alpha,
    resolve_prior_weight,
    resolve_ts_sigma,
    run_template_bandits_experiment_gpu,
)
from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module


@dataclass
class Scenario:
    """Single GPU-enabled experiment definition."""

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


class _StdoutTee(io.TextIOBase):
    """Mirror writes to stdout while capturing scenario logs."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated batch runner for Chapter 6 template bandit scenarios."
    )
    parser.add_argument("--n-static", type=int, default=1_000)
    parser.add_argument("--n-bandit", type=int, default=20_000)
    parser.add_argument("--world-seed-base", type=int, default=2025_0701)
    parser.add_argument("--bandit-seed-base", type=int, default=2025_0801)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/book/ch06/data"),
    )
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device identifier (e.g. 'cuda', 'cuda:0', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="bandit_matrix_gpu",
    )
    parser.add_argument(
        "--stream-plan",
        action="store_true",
        help="Print scenario plan with resolved hyperparameters.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent scenarios (default 1 for GPU safety).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution regardless of --max-workers.",
    )
    parser.add_argument(
        "--stream-logs",
        action="store_true",
        help="Mirror stdout while capturing it into the artifact.",
    )
    parser.add_argument(
        "--show-volume",
        action="store_true",
        help="Include Orders column in summary tables.",
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
    return 100.0 * delta / base if base != 0.0 else 0.0


def _resolve_metrics(mapping: Dict, key: int) -> Dict[str, float]:
    if key in mapping:
        return mapping[key]
    s_key = str(key)
    if s_key in mapping:
        return mapping[s_key]
    raise KeyError(f"Missing metrics for template {key}")


def _segment_metrics(
    segments: Dict[str, Dict[str, float]],
    name: str,
    fallback: Dict[str, float],
) -> Dict[str, float]:
    return segments.get(name, fallback)


def _print_static_table(results: Dict[str, Any], show_volume: bool) -> None:
    print("\nStatic templates (per-episode averages):")
    if show_volume:
        print(
            f"{'ID':>2s}  {'Template':15s}  {'Reward':>10s}  "
            f"{'GMV':>10s}  {'CM2':>10s}  {'Orders':>10s}"
        )
    else:
        print(
            f"{'ID':>2s}  {'Template':15s}  {'Reward':>10s}  "
            f"{'GMV':>10s}  {'CM2':>10s}"
        )
    static_results = results["static_results"]
    for template in results["templates"]:
        tid = template["id"]
        metrics = _resolve_metrics(static_results, tid)
        if show_volume:
            print(
                f"{tid:2d}  {template['name']:15s}  {metrics['reward']:10.2f}  "
                f"{metrics['gmv']:10.2f}  {metrics['cm2']:10.2f}  "
                f"{metrics['orders']:10.2f}"
            )
        else:
            print(
                f"{tid:2d}  {template['name']:15s}  {metrics['reward']:10.2f}  "
                f"{metrics['gmv']:10.2f}  {metrics['cm2']:10.2f}"
            )


def _print_policy_summary(results: Dict[str, Any], show_volume: bool) -> None:
    best = results["static_best"]["result"]
    lin = results["linucb"]["global"]
    ts = results["ts"]["global"]
    rows = [
        (f"Static-{results['static_best']['name']}", best),
        ("LinUCB", lin),
        ("ThompsonSampling", ts),
    ]
    print("\nSummary (per-episode averages):")
    if show_volume:
        print(
            f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  {'CM2':>10s}  "
            f"{'Orders':>10s}  {'ΔGMV vs static':>14s}"
        )
    else:
        print(
            f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  "
            f"{'CM2':>10s}  {'ΔGMV vs static':>14s}"
        )
    for label, metrics in rows:
        delta = _pct(metrics["gmv"] - best["gmv"], best["gmv"])
        if show_volume:
            print(
                f"{label:18s}  {metrics['reward']:10.2f}  {metrics['gmv']:10.2f}  "
                f"{metrics['cm2']:10.2f}  {metrics['orders']:10.2f}  "
                f"{delta:+13.2f}%"
            )
        else:
            print(
                f"{label:18s}  {metrics['reward']:10.2f}  {metrics['gmv']:10.2f}  "
                f"{metrics['cm2']:10.2f}  {delta:+13.2f}%"
            )


def _print_segment_summary(
    *,
    results: Dict[str, Any],
    segment_order: Sequence[str],
    show_volume: bool,
) -> None:
    static_segments = results["static_best"]["segments"]
    lin_segments = results["linucb"]["segments"]
    ts_segments = results["ts"]["segments"]
    static_global = results["static_best"]["result"]

    if show_volume:
        print("\nPer-segment GMV & Orders (static best vs bandits):")
        print(
            f"{'Segment':15s}  {'Static GMV':>10s}  {'LinUCB GMV':>10s}  "
            f"{'TS GMV':>10s}  {'Static Orders':>14s}  {'LinUCB Orders':>14s}  "
            f"{'TS Orders':>11s}  {'Lin GMV Δ%':>12s}  {'TS GMV Δ%':>11s}  "
            f"{'Lin Orders Δ%':>15s}  {'TS Orders Δ%':>14s}"
        )
    else:
        print("\nPer-segment GMV (static best vs bandits):")
        print(
            f"{'Segment':15s}  {'Static GMV':>10s}  {'LinUCB GMV':>10s}  "
            f"{'TS GMV':>10s}  {'LinUCB Δ%':>11s}  {'TS Δ%':>8s}"
        )

    for seg in segment_order:
        s = _segment_metrics(static_segments, seg, static_global)
        l = _segment_metrics(lin_segments, seg, static_global)
        t = _segment_metrics(ts_segments, seg, static_global)
        lin_delta = _pct(l["gmv"] - s["gmv"], s["gmv"])
        ts_delta = _pct(t["gmv"] - s["gmv"], s["gmv"])
        if show_volume:
            lin_orders_delta = _pct(l["orders"] - s["orders"], s["orders"]) if s["orders"] else 0.0
            ts_orders_delta = _pct(t["orders"] - s["orders"], s["orders"]) if s["orders"] else 0.0
            print(
                f"{seg:15s}  {s['gmv']:10.2f}  {l['gmv']:10.2f}  {t['gmv']:10.2f}  "
                f"{s['orders']:14.2f}  {l['orders']:14.2f}  {t['orders']:11.2f}  "
                f"{lin_delta:+12.2f}%  {ts_delta:+11.2f}%  "
                f"{lin_orders_delta:+15.2f}%  {ts_orders_delta:+14.2f}%"
            )
        else:
            print(
                f"{seg:15s}  {s['gmv']:10.2f}  {l['gmv']:10.2f}  {t['gmv']:10.2f}  "
                f"{lin_delta:+10.2f}%  {ts_delta:+7.2f}%"
            )


def _print_template_freqs(results: Dict[str, Any]) -> None:
    lin_freqs = results["linucb"]["diagnostics"]["template_freqs"]
    ts_freqs = results["ts"]["diagnostics"]["template_freqs"]
    template_names = [t["name"] for t in results["templates"]]

    print("\nTemplate selection frequencies — LinUCB:")
    for idx, (name, freq) in enumerate(zip(template_names, lin_freqs)):
        print(f"  {idx:2d} {name:15s}: {100.0 * freq:6.2f}%")
    print("\nTemplate selection frequencies — ThompsonSampling:")
    for idx, (name, freq) in enumerate(zip(template_names, ts_freqs)):
        print(f"  {idx:2d} {name:15s}: {100.0 * freq:6.2f}%")


def _print_full_summary(
    *,
    label: str,
    results: Dict[str, Any],
    segment_order: Sequence[str],
    show_volume: bool,
) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    _print_static_table(results, show_volume)
    _print_policy_summary(results, show_volume)
    _print_segment_summary(
        results=results,
        segment_order=segment_order,
        show_volume=show_volume,
    )
    _print_template_freqs(results)


def _summarize(results: Dict[str, Any]) -> Dict[str, float]:
    static_res = results["static_best"]["result"]
    lin_res = results["linucb"]["global"]
    ts_res = results["ts"]["global"]
    static_gmv = static_res["gmv"]
    return {
        "static_gmv": static_gmv,
        "linucb_gmv": lin_res["gmv"],
        "ts_gmv": ts_res["gmv"],
        "linucb_delta_pct": _pct(lin_res["gmv"] - static_gmv, static_gmv),
        "ts_delta_pct": _pct(ts_res["gmv"] - static_gmv, static_gmv),
    }


def _run_scenario(
    scenario: Scenario,
    *,
    batch_size: int,
    device: Optional[str],
    show_volume: bool,
    stream_logs: bool,
) -> Dict[str, Any]:
    cfg = SimulatorConfig(seed=scenario.world_seed)
    rng = np.random.default_rng(cfg.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    prior, lin_alpha, ts_sigma = scenario.resolve_hparams()
    rich_cfg = scenario.build_rich_config()

    def _execute() -> Dict[str, Any]:
        return run_template_bandits_experiment_gpu(
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
            batch_size=batch_size,
            device=None if device == "auto" else device,
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
            "batch_size": batch_size,
            "requested_device": device,
            "resolved_device": results["config"].get("device"),
        }
    )
    _print_full_summary(
        label=f"Scenario Summary — {scenario.name}",
        results=results,
        segment_order=cfg.users.segments,
        show_volume=show_volume,
    )
    return {
        "scenario": metadata,
        "results": results,
        "summary": _summarize(results),
        "stdout": buffer.getvalue(),
    }


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
    *,
    scenarios: List[Scenario],
    batch_size: int,
    device: Optional[str],
    show_volume: bool,
    max_workers: int,
    sequential: bool,
    stream_logs: bool,
) -> List[Dict[str, Any]]:
    if sequential or max_workers <= 1:
        outputs = []
        total = len(scenarios)
        for idx, scenario in enumerate(scenarios, start=1):
            print(f"\n[{idx}/{total}] Running scenario '{scenario.name}' on {device}...")
            result = _run_scenario(
                scenario,
                batch_size=batch_size,
                device=device,
                show_volume=show_volume,
                stream_logs=stream_logs,
            )
            outputs.append(result)
            summary = result["summary"]
            print(
                f"✓ {scenario.name}: LinUCB Δ={summary['linucb_delta_pct']:+5.2f}% "
                f"TS Δ={summary['ts_delta_pct']:+5.2f}%"
            )
        return outputs

    workers = min(max_workers, len(scenarios))
    print(
        f"\nRunning {len(scenarios)} scenarios with {workers} worker threads "
        f"(device={device})..."
    )
    outputs_map: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_scenario,
                scenario,
                batch_size=batch_size,
                device=device,
                show_volume=show_volume,
                stream_logs=stream_logs,
            ): scenario
            for scenario in scenarios
        }
        completed = 0
        for future in as_completed(futures):
            scenario = futures[future]
            try:
                result = future.result()
                outputs_map[scenario.name] = result
                completed += 1
                summary = result["summary"]
                print(
                    f"✓ [{completed}/{len(scenarios)}] {scenario.name}: "
                    f"LinUCB Δ={summary['linucb_delta_pct']:+5.2f}% "
                    f"TS Δ={summary['ts_delta_pct']:+5.2f}%"
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
    if args.stream_plan:
        _print_plan(scenarios)

    outputs = run_scenarios(
        scenarios=scenarios,
        batch_size=args.batch_size,
        device=args.device,
        show_volume=args.show_volume,
        max_workers=args.max_workers,
        sequential=args.sequential,
        stream_logs=args.stream_logs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"{args.filename_prefix}_{timestamp}.json"
    effective_workers = None
    if not args.sequential and args.max_workers > 1:
        effective_workers = min(args.max_workers, len(scenarios))
    artifact = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_file": str(output_path),
        "parallel_workers": effective_workers,
        "scenarios": outputs,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved batch results to {output_path}")


if __name__ == "__main__":
    main()
