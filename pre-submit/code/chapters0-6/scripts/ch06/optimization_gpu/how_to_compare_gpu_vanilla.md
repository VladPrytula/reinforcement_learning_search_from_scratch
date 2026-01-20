run_bandit_matrix_gpu.py doesn’t accept --scenario-file (same for the CPU runner); both scripts bake in the Chapter 6
  default scenario set via build_default_scenarios(...). To do the “small-count” parity check you can simply override
  the episode counts/seed bases and send each runner’s artifact to a separate directory:

  ART_ROOT=/tmp/ch06_parity && mkdir -p "$ART_ROOT/cpu" "$ART_ROOT/gpu"

  # 1) Canonical CPU run (500/500 episodes)
  ```
  python scripts/ch06/run_bandit_matrix.py \
    --n-static 500 --n-bandit 500 \
    --world-seed-base 20250319 --bandit-seed-base 20250319 \
    --output-dir "$ART_ROOT/cpu" \
    --filename-prefix parity_cpu \
    --sequential --max-workers 1
```

  # 2) GPU run with the same knobs (force single worker for determinism)
  ```
  python scripts/ch06/optimization_gpu/run_bandit_matrix_gpu.py \
    --n-static 500 --n-bandit 500 \
    --world-seed-base 20250319 --bandit-seed-base 20250319 \
    --output-dir "$ART_ROOT/gpu" \
    --filename-prefix parity_gpu \
    --batch-size 512 \
    --device auto \
    --sequential --max-workers 1
```

  Each command writes a single JSON artifact (e.g., `/tmp/ch06_parity/cpu/parity_cpu_*.json`). Diff them with jq:

``
  CPU_JSON=$(ls -t "$ART_ROOT/cpu"/parity_cpu_*.json | head -n1)
  GPU_JSON=$(ls -t "$ART_ROOT/gpu"/parity_gpu_*.json | head -n1)
``
``
  jq '.scenarios[].summary' "$CPU_JSON" > /tmp/cpu.txt
  jq '.scenarios[].summary' "$GPU_JSON" > /tmp/gpu.txt
  diff -u /tmp/cpu.txt /tmp/gpu.txt
``

(You can also inspect a specific scenario: `jq '.scenarios[] | {name: .scenario.name, lin:.summary.linucb_gmv,
ts:.summary.ts_gmv}' "$CPU_JSON"` and the same for the GPU file.)

Once the 500‑episode comparison looks good, rerun both scripts with your production `--n-static/--n-bandit` settings and
  desired `worker counts/output` locations.
