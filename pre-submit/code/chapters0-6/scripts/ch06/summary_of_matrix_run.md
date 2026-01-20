CPU/classic artifact docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json and ignored the GPU runs.

  - simple_baseline — with only segment + query features the scenario is intentionally a failure mode, so the static Premium template’s GMV 7.26
    beats LinUCB 7.10 (‑2.2 %) and TS 5.02 (‑30.8 %), confirming that bandits over-explore cheaper templates whenever the context is too weak (docs/
    book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:8, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:9, docs/book/drafts/
    ch06/data/bandit_matrix_20251118T031855Z.json:287, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:288, docs/book/drafts/ch06/
    data/bandit_matrix_20251118T031855Z.json:289, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:290, docs/book/drafts/ch06/data/
    bandit_matrix_20251118T031855Z.json:291).
  - rich_oracle_raw — even though oracle latents are available, running them “raw” without blending keeps posterior variance high; both LinUCB
    7.20 (‑8.7 %) and TS 7.11 (‑9.8 %) trail the static Premium template’s 7.89 GMV, which matches the expectation that unregularized features
    overfit and miss the high-value premium customers (docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:297, docs/book/drafts/
    ch06/data/bandit_matrix_20251118T031855Z.json:298, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:576, docs/book/drafts/ch06/
    data/bandit_matrix_20251118T031855Z.json:577, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:578, docs/book/drafts/ch06/data/
    bandit_matrix_20251118T031855Z.json:579, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:580).
  - rich_oracle_blend — once oracle latents are blended with segment priors and shrinkage, LinUCB jumps to 8.91 GMV (+15.0 %) while TS stays just under
    static at 7.66 (‑1.2 %); optimism with tuned α helps LinUCB exploit the denser signal whereas TS retains more exploratory noise at σ=1.0 (docs/
    book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:596, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:597, docs/book/
    drafts/ch06/data/bandit_matrix_20251118T031855Z.json:882, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:883, docs/book/drafts/
    ch06/data/bandit_matrix_20251118T031855Z.json:884, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:885, docs/book/drafts/ch06/data/
    bandit_matrix_20251118T031855Z.json:886).
  - rich_oracle_quantized — quantizing the oracle latents (a proxy for production estimates) stabilizes the signal and both bandits clear the static
    Premium template: LinUCB up to 9.05 GMV (+28.1 %) and TS to 9.26 (+31.1 %), while also delivering higher shared KPIs (reward and CM2) because both
    policies converge on higher-value premium/pl-lover segments (docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:613, docs/book/drafts/
    ch06/data/bandit_matrix_20251118T031855Z.json:1188, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:1189, docs/book/drafts/ch06/
    data/bandit_matrix_20251118T031855Z.json:1190, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:1191, docs/book/drafts/ch06/data/
    bandit_matrix_20251118T031855Z.json:1192). This is the only run in the CPU batch where both LinUCB and TS beat the static business KPI.
  - rich_estimated — using noisy estimated latents (production-style) keeps TS ahead of static at 8.94 GMV (+26.2 %) but LinUCB slips to 6.94 (‑2.0 %),
    suggesting its exploration bonus (α=0.85) is too low once the features are noisy, while TS’s sampling tolerates the estimation error better (docs/
    book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:629, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:1477, docs/book/drafts/
    ch06/data/bandit_matrix_20251118T031855Z.json:1478, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:1479, docs/book/drafts/ch06/data/
    bandit_matrix_20251118T031855Z.json:1480, docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json:1481).

  Natural next steps: 1) re-run the “rich_estimated” scenario after tuning LinUCB’s α (or adding posterior shrinkage) to see if it can match TS; 2) capture
  the quantized run in the Chapter 6 draft as the canonical example where both contextual policies exceed the static template so readers can see what “good”
  looks like.