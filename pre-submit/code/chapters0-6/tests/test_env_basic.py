import numpy as np

from zoosim.core import config
from zoosim.envs import GymZooplusEnv, ZooplusSearchEnv


def test_core_env_step_runs():
    cfg = config.load_default_config()
    env = ZooplusSearchEnv(cfg, seed=123)
    state = env.reset()
    assert "phi_cat" in state and state["phi_cat"].shape[0] == len(cfg.catalog.categories)

    action = np.zeros(cfg.action.feature_dim)
    next_state, reward, done, info = env.step(action)
    assert next_state is None
    assert isinstance(reward, float)
    assert done is True
    assert "reward_details" in info
    assert len(info["clicks"]) == cfg.top_k


def test_gym_env_wrapper():
    gym_env = GymZooplusEnv(seed=99)
    obs, info = gym_env.reset()
    assert gym_env.observation_space.contains(obs)

    action = gym_env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = gym_env.step(action)
    assert gym_env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert "raw_state" in info2
