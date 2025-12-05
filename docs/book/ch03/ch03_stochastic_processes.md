# Chapter 3 — Stochastic Processes and the Bellman View

*Vlad Prytula*

## Motivation
The contextual bandit of Chapter 1 is the special case of the Bellman optimality equation with `\(\gamma=0\)`; see [EQ-1.21] and the bandit value equations [EQ-1.8–1.10]. To treat multi-step sessions (satisfaction dynamics, abandonment, cart building), we need the language of **stochastic processes** (filtrations, stopping times) and the connection to the **Bellman operator** for `\(\gamma>0\)`.

## Roadmap
- 3.1 Satisfaction process `\(S_t\)` with a stopping time `\(\tau\)` for abandonment; optional stopping conditions
- 3.2 Mapping to simulator parameters (`BehaviorConfig`): `pos_bias`, `satisfaction_decay`, `abandonment_threshold`
- 3.3 From processes to Bellman backups: deriving value iteration in the single-session limit

## Cross-References
- Bandit Bellman (γ=0): [EQ-1.21]; value definitions: [EQ-1.8–1.10]
- Reward components and aggregation: `zoosim/dynamics/reward.py:1`, `zoosim/core/config.py:193`

!!! tip "Production Checklist (Chapter 3)"
    - Validate position-bias arrays shapes vs. `top_k` (`SimulatorConfig.top_k`).
    - Calibrate abandonment and satisfaction parameters in `BehaviorConfig` with seeds.
    - Keep session RNGs deterministic per episode for reproducibility.
    - Enforce action bounds each step if using multi-step variants (match `action.a_max`).

TODO: flesh out proofs and include diagrams.
