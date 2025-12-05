
import numpy as np
import pytest
from typing import Dict, Any
from zoosim.evaluation.ope import (
    ips_estimate, snips_estimate, pdis_estimate, dr_estimate, fqe,
    Trajectory, Transition, Policy, QFunction
)

# --- Mocks for Testing ---

class MockPolicy(Policy):
    def __init__(self, target_action: float, sigma: float = 1.0):
        self.target_action = np.array([target_action])
        self.sigma = sigma

    def act(self, state: Dict[str, Any]) -> np.ndarray:
        return self.target_action

    def prob(self, action: np.ndarray, state: Dict[str, Any]) -> float:
        # Simple Gaussian density
        diff = action - self.target_action
        return float(np.exp(-0.5 * (diff / self.sigma)**2) / (np.sqrt(2 * np.pi) * self.sigma))

class MockQFunction(QFunction):
    def __init__(self, value_fixed: float):
        self.value_fixed = value_fixed
        
    def value(self, state: Dict[str, Any], policy: Policy) -> float:
        return self.value_fixed
        
    def q(self, state: Dict[str, Any], action: np.ndarray) -> float:
        return self.value_fixed
        
    def forward(self, states, actions):
        import torch
        return torch.ones(states.shape[0]) * self.value_fixed

    def parameters(self):
        import torch
        return iter([torch.nn.Parameter(torch.tensor(0.0))])

def create_dummy_trajectory(action_val: float, reward: float, propensity: float) -> Trajectory:
    trans = Transition(
        state={"features": np.zeros(1)},
        action=np.array([action_val]),
        reward=reward,
        next_state={"features": np.zeros(1)},
        done=True,
        propensity=propensity
    )
    return Trajectory(transitions=[trans], return_=reward, length=1)

def test_ips_unbiased():
    """Test if IPS recovers true value when weights are correct."""
    # Setup:
    # Behavior policy: Uniform[-1, 1] -> p(a) = 0.5
    # Target policy: Uniform[-1, 1] -> p(a) = 0.5
    # Reward: R(a) = a
    # Expected return for target: E[a] = 0

    # We simulate data from Behavior
    # Since policies are identical, weights should be 1.0.
    # IPS should just be average return.

    # Fixed seed for reproducibility (deterministic tests per CLAUDE.md)
    rng = np.random.default_rng(seed=42)

    pi_e = MockPolicy(0.0, sigma=1000.0) # Very wide, essentially flat
    # Actually let's just force prob to be 0.5 for testing
    pi_e.prob = lambda a, s: 0.5

    dataset = []
    for i in range(100):
        a = rng.uniform(-1, 1)
        r = a
        traj = create_dummy_trajectory(a, r, propensity=0.5)
        dataset.append(traj)
        
    est, _ = ips_estimate(dataset, pi_e)
    
    # Average of Uniform[-1, 1] is 0.
    # SE = 1/sqrt(12) / sqrt(100) ~= 0.28 / 10 = 0.028
    assert np.abs(est) < 0.1 # Sufficiently close

def test_importance_weighting():
    """Test weighting logic explicitly."""
    # Behavior: always picks 1.0 with prob 1.0
    # Target: picks 1.0 with prob 0.5
    # Weight should be 0.5
    
    pi_e = MockPolicy(1.0)
    pi_e.prob = lambda a, s: 0.5
    
    traj = create_dummy_trajectory(1.0, 10.0, propensity=1.0)
    dataset = [traj]
    
    est, diag = ips_estimate(dataset, pi_e)
    
    # E[w * G] = 0.5 * 10 = 5.0
    assert est == 5.0
    assert diag["weights_mean"] == 0.5

def test_snips_normalization():
    """Test SNIPS normalization."""
    # Two trajectories.
    # T1: w=0.5, G=10
    # T2: w=1.5, G=20
    # SNIPS = (0.5*10 + 1.5*20) / (0.5 + 1.5) = (5 + 30) / 2 = 17.5
    
    pi_e = MockPolicy(0.0)
    
    # Hack probs to get desired weights
    def prob_fn(a, s):
        if a[0] == 1.0: return 0.5 # T1 numerator
        if a[0] == 2.0: return 1.5 # T2 numerator
        return 0.0
    pi_e.prob = prob_fn
    
    traj1 = create_dummy_trajectory(1.0, 10.0, propensity=1.0)
    traj2 = create_dummy_trajectory(2.0, 20.0, propensity=1.0)
    
    est, _ = snips_estimate([traj1, traj2], pi_e)
    
    assert np.isclose(est, 17.5)

def test_dr_perfect_model():
    """Test DR with a perfect model."""
    # If Q(s,a) is perfect, DR should have 0 variance and exact answer.
    # V(s0) = True Value.
    # Correction term should be mean-zero or exactly cancel if deterministic.
    
    pi_e = MockPolicy(0.0)
    pi_e.prob = lambda a, s: 1.0
    
    # True value is 10.0
    q_model = MockQFunction(10.0)
    
    # Trajectory: single step, reward 10.0
    # V(s0) = 10.0
    # Q(s0, a0) = 10.0
    # V(s1) = 10.0 (but done, so 0.0)
    # TD error = r + gamma*V' - Q = 10 + 0 - 10 = 0.
    # Estimate = V(s0) + 0 = 10.
    
    traj = create_dummy_trajectory(0.0, 10.0, propensity=1.0)
    
    est, _ = dr_estimate([traj], pi_e, q_model, gamma=0.0) # gamma 0 for simplicity
    
    assert est == 10.0

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
