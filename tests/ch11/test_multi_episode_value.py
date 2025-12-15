import numpy as np

from zoosim.multi_episode.session_env import MultiSessionEnv, SessionMDPState

from scripts.ch11.utils.trajectory import MultiEpisodeTrajectory, collect_trajectory


def _zero_policy(state: SessionMDPState) -> np.ndarray:
    """Baseline policy: no boost."""
    # The action dimension is fixed by the simulator config (feature_dim).
    return np.zeros(10, dtype=float)


def test_discounted_value_matches_manual_sum() -> None:
    """Discounted value in the trajectory should match manual computation."""
    env = MultiSessionEnv(seed=2025)
    gamma = 0.95

    traj = collect_trajectory(env, _zero_policy, gamma=gamma, max_sessions=5)
    assert isinstance(traj, MultiEpisodeTrajectory)
    assert traj.n_sessions == len(traj.session_rewards)

    manual = 0.0
    discount = 1.0
    for r in traj.session_rewards:
        manual += discount * r
        discount *= gamma

    assert np.isclose(traj.discounted_value, manual)


def test_collect_trajectory_seed_deterministic() -> None:
    """Trajectories should be reproducible for the same environment seed."""
    gamma = 0.9

    env1 = MultiSessionEnv(seed=777)
    env2 = MultiSessionEnv(seed=777)

    traj1 = collect_trajectory(env1, _zero_policy, gamma=gamma, max_sessions=10)
    traj2 = collect_trajectory(env2, _zero_policy, gamma=gamma, max_sessions=10)

    assert traj1.session_rewards == traj2.session_rewards
    assert traj1.session_clicks == traj2.session_clicks
    assert np.isclose(traj1.discounted_value, traj2.discounted_value)

