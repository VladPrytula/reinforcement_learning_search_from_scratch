import numpy as np

from zoosim.core.config import load_default_config
from zoosim.multi_episode.session_env import MultiSessionEnv, SessionMDPState


def test_reset_returns_valid_state() -> None:
    """Reset should return a well-formed SessionMDPState."""
    cfg = load_default_config()
    env = MultiSessionEnv(cfg=cfg, seed=123)

    state = env.reset()

    assert isinstance(state, SessionMDPState)
    assert state.t == 0
    assert isinstance(state.user_segment, str)
    assert isinstance(state.query_type, str)
    # State carries a category embedding; allow list/tuple/ndarray.
    assert isinstance(state.phi_cat, (list, tuple, np.ndarray))
    assert state.last_clicks == 0
    assert state.last_satisfaction == 0.0


def test_step_returns_correct_types_and_info() -> None:
    """Step should return expected types and core info keys."""
    env = MultiSessionEnv(seed=42)
    env.reset()

    action = np.zeros(env.cfg.action.feature_dim)
    next_state, reward, done, info = env.step(action)

    assert isinstance(next_state, SessionMDPState)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Core diagnostics promised by the chapter text
    for key in [
        "reward_details",
        "satisfaction",
        "ranking",
        "clicks",
        "buys",
        "features",
        "p_return",
        "returned",
    ]:
        assert key in info


def test_same_user_persists_until_churn() -> None:
    """User segment should remain stable across sessions until churn."""
    env = MultiSessionEnv(seed=7)
    state = env.reset()
    initial_segment = state.user_segment

    action = np.zeros(env.cfg.action.feature_dim)
    for _ in range(50):
        next_state, _, done, _ = env.step(action)
        if done:
            # Terminal state uses sentinel values
            assert next_state.query_type == "terminal"
            assert next_state.phi_cat == []
            break
        assert next_state.user_segment == initial_segment


def test_deterministic_trajectories_with_seed() -> None:
    """Same seed and policy should yield identical reward sequences."""

    def run_episode(seed: int) -> list[float]:
        env = MultiSessionEnv(seed=seed)
        env.reset()
        rewards: list[float] = []
        action = np.zeros(env.cfg.action.feature_dim)
        for _ in range(10):
            _, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        return rewards

    r1 = run_episode(1234)
    r2 = run_episode(1234)
    assert r1 == r2
