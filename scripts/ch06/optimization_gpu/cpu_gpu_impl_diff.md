
  1. Different user/query/relevance pipeline (CPU uses Zoosim, GPU reimplements it)

  CPU “ground truth” path (one episode):

  - User sampling:
      - sample_user(config=cfg, rng=rng) in zoosim/world/users.py:26-44
  - Query sampling:
      - sample_query(user=user, config=cfg, rng=rng) in zoosim/world/queries.py:34-63
  - Base relevance:
      - relevance.batch_base_scores(...) in zoosim/ranking/relevance.py:32-45
  - Per-template episode:
      - _simulate_episode_for_template(...) in scripts/ch06/template_bandits_demo.py:442-486
          - Computes base_scores + template.apply(products)
          - Calls behavior.simulate_session(...) (zoosim/dynamics/behavior.py)
          - Calls reward.compute_reward(...) (zoosim/dynamics/reward.py)

  Bandit loop and static baselines both call that exact stack.

  GPU pipeline does not call those helpers. It reimplements the whole sampling + relevance stack in Torch:

  - User latents:
      - _sample_segments, _sample_theta, _sample_theta_emb in
        scripts/ch06/optimization_gpu/template_bandits_gpu.py:320-380
  - Query types/intents/embeddings:
      - _sample_query_types, _sample_query_intents, _sample_query_embeddings in
        scripts/ch06/optimization_gpu/template_bandits_gpu.py:360-409
  - Base relevance:
      - _build_lexical_matrix and _compute_base_scores in
        scripts/ch06/optimization_gpu/template_bandits_gpu.py:200-239 and 620-639
  - Session simulation:
      - _simulate_sessions in
        scripts/ch06/optimization_gpu/template_bandits_gpu.py:640-736

  Even though you aimed to mirror the math, you’re no longer literally executing:

  - sample_user
  - sample_query
  - relevance.base_score

  That alone is enough to change the MDP slightly. With 50k episodes, “slightly different” is enough to give the kind of GMV shifts you’re seeing, especially in sensitive regimes
  like rich_estimated.

  2. Lexical relevance is implemented differently (this is a big one)

  CPU lexical component:

  - In zoosim/ranking/relevance.py:19-31:
      - Query.tokens includes:
          - Tokens from the chosen intent category name (intent_category.split("_"))
          - The query_type string ("category", "search", etc.)
          - Two random noise tokens tok_<id> per query (_sample_tokens in zoosim/world/queries.py:24-31)
      - _lexical_component does:
          - query_tokens = set(query.tokens)
          - prod_tokens = set(product.category.split("_"))
          - Overlap = size of intersection; score = log1p(overlap)

  GPU lexical component:

  - In _build_lexical_matrix (template_bandits_gpu.py:200-239):
      - Precomputes a matrix of shape (n_categories, n_products) with:
          - cat_tokens = set(cat.split("_")) for each catalog category
          - prod_tokens = set(product.category.split("_"))
          - Cell = log1p(len(cat_tokens & prod_tokens))
  - In _compute_base_scores (template_bandits_gpu.py:620-639):
      - For each episode, selects a single row: lexical = world.lexical_matrix[query_intent_idx]
      - No query_type token, no random tok_* tokens.

  So, on CPU:

  - Lexical relevance depends on:
      - The intent category
      - The query type
      - Random extra tokens

  On GPU:

  - Lexical relevance depends only on:
      - The intent category (via query_intent_idx)

  Even if most of the time the extra tokens don’t change overlap, that’s still a systematic modeling change. Since:

  - Rich features use aggregates over base top‑k (context_features_rich / _compute_base_topk_aggregates).
  - Bandits depend heavily on those aggregates in rich, rich_est regimes.

  …this lexical difference is a major reason why:

  - Static GMV moves notably in some scenarios (e.g. rich_oracle_quantized static_gmv +16% on GPU vs CPU).
  - Bandit policies behave differently for the same hyperparams (especially rich_estimated, where CPU says “LinUCB/TS ≈ static or slightly worse” and GPU says “LinUCB/TS +25%”).

  3. GPU never calls the CPU sampling helpers, so RNG flows differ

  On CPU:

  - One NumPy RNG rng = np.random.default_rng(seed) drives:
      - Segment choice, θ_price, θ_pl, θ_cat
      - Query intent, query type, noise in embeddings
      - Relevance noise
      - Behavior noise in simulate_session

  On GPU:

  - You use:
      - One torch.Generator for segments, θ_price, θ_pl, θ_emb, query types/intents, relevance noise, behavior noise.
      - One np.random.Generator for the gamma draws in _sample_theta and for some other pieces.

  Even with the same “base seeds”:

  - The sequence of random draws, and how they’re split between NumPy and Torch, diverges from the CPU path.
  - Mathematically the distributions are the same, but episode‑by‑episode trajectories are not comparable one‑for‑one.

  If you had only this RNG split and everything else identical, you’d expect small differences that vanish as n→∞, not the 20–30% shifts you’re observing. But combined with the
  lexical and sampling differences, the MDP is effectively different.

  4. Static evaluation and bandit loops are structurally similar and mostly fine

  These parts actually match pretty well:

  - CPU static:
      - _run_static_templates in template_bandits_demo.py:486-539
          - For each episode: sample user, query, base_scores.
          - For each template: call _simulate_episode_for_template.
  - GPU static:
      - _compute_static_results in template_bandits_gpu.py:840-869
          - Calls _simulate_episode_batch with feature_mode="simple" and gets all templates in one shot.
  - CPU bandit loop:
      - _run_bandit_policy in template_bandits_demo.py:743-828
          - One episode at a time: sample user/query, compute base_scores, build features, select action, simulate that template, update policy.
  - GPU bandit loop:
      - _run_policy_interactive in template_bandits_gpu.py:900-961
          - For each batch: _simulate_episode_batch builds features and metrics for all templates, one action is chosen per episode, only that action’s reward is used for update.

  You did a good job preserving the logic here. The big non‑1‑to‑1 changes are in how the inputs (user/query/relevance) are generated, not in how the bandits update.

  5. Why the differences show up so strongly in your results

  Given these concrete differences, your observed artifacts make sense:

  - rich_oracle_raw:
      - Static and LinUCB GMV are very close between CPU and GPU → the GPU simulator is “close enough” in that simpler setting.
  - rich_oracle_blend, rich_oracle_quantized, rich_estimated:
      - These regimes are more sensitive:
          - Estimated/regularized user latents
          - Rich aggregates over base top‑k
      - Here, the lexical simplifications + slightly different sampling pipeline + different RNG flow produce a world where:
          - Static GMV and bandit GMV can both shift.
          - The relative ranking (bandits vs static) can flip, as you saw in rich_estimated.

  So the core answer to “why are my results different?” is:

  - The GPU code isn’t actually a minimal line‑for‑line translation of the CPU path.
  - It reimplements the user/query/relevance side of the simulator in Torch (especially lexical relevance), and that makes the simulated environment meaningfully different.

  6. What to do if you really want 1‑to‑1 parity

  If you truly want “same results up to sampling noise” between CPU and GPU, you have two options:

  - A. Make the GPU path call the CPU sampling + relevance helpers:
      - In template_bandits_gpu.py:
          - Instead of _sample_segments / _sample_theta / _sample_query_types / _sample_query_intents / _sample_query_embeddings / _compute_base_scores, call:
              - sample_user
              - sample_query
              - relevance.batch_base_scores
          - Only move the heavy parts (base score → boost → ranking → reward aggregation) onto the GPU.
      - This will be slower than your current fully‑batched approach but will give you true 1‑to‑1 behavior.
  - B. If you want to keep the fully batched GPU sampler:
      - Accept that it’s an approximate simulator and calibrate it against CPU for each regime before trusting the results.
      - If you still want closer behavior, the single biggest change to make is:
          - Replace _build_lexical_matrix / _compute_base_scores with something that uses the same lexical logic as relevance.base_score (i.e., use Query.tokens including
            query_type and random tokens), even if partially.

  Right now, the GPU implementation is consistent but it is not a literal port of:

  - sample_user
  - sample_query
  - relevance.base_score

  That’s the root reason your CPU and GPU results differ.