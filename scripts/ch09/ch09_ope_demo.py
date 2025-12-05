import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from zoosim.core.config import SimulatorConfig
from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.policies.q_ensemble import QEnsemblePolicy as BaseQEnsemblePolicy
from zoosim.policies.reinforce import REINFORCEAgent
from zoosim.evaluation.ope import (
    epsilon_greedy_logger, ips_estimate, snips_estimate, pdis_estimate, dr_estimate, fqe,
    QEnsemblePolicy as OPEQEnsemblePolicy, 
    REINFORCEPolicy as OPEREINFORCEPolicy
)

def main():
    print("--- Chapter 9: Off-Policy Evaluation Demo ---")
    
    # 0. Setup
    cfg = SimulatorConfig()
    # Ensure consistent dimensions
    state_dim = 17 # Default in SearchEnv features
    action_dim = 10 # Default boost dim
    
    # Create Env
    # Using rich_features=True to get the vector observation
    env = GymZooplusEnv(cfg, seed=42, rich_features=True)
    
    # Check actual dims
    obs, _ = env.reset()
    if isinstance(obs, dict):
        real_state_dim = obs["features"].shape[0]
    else:
        real_state_dim = obs.shape[0]
        
    print(f"State Dimension: {real_state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    if real_state_dim != state_dim:
        print(f"Adjusting state dim to {real_state_dim}")
        state_dim = real_state_dim

    # 1. Behavior Policy (Untrained Q-Ensemble)
    print("\n1. Initializing Behavior Policy (random Q-Ensemble)...")
    # Base policy from Ch7
    pi_b_base = BaseQEnsemblePolicy(
        state_dim=state_dim, 
        action_dim=action_dim, 
        a_max=1.0,
        ucb_beta=0.0 # Deterministic argmax
    )
    # Wrap for OPE (adds probability calculation)
    # We add small noise to behavior policy itself (Gaussian)
    pi_b_ope = OPEQEnsemblePolicy(pi_b_base, exploration_noise=0.1)

    # 2. Collect Logged Data (with extra epsilon-greedy exploration)
    print("2. Collecting logged data (epsilon=0.1)...")
    # This mixes the Gaussian policy with Uniform random actions
    logged_data = epsilon_greedy_logger(
        env, 
        pi_b_ope, 
        epsilon=0.1, 
        num_episodes=50, # Small for demo speed
        seed=42
    )

    avg_return = np.mean([traj.return_ for traj in logged_data])
    print(f"Collected {len(logged_data)} trajectories.")
    print(f"Average Behavior Return: {avg_return:.2f}")

    # 3. Evaluation Policy (Untrained REINFORCE)
    print("\n3. Initializing Evaluation Policy (random REINFORCE)...")
    # Base agent from Ch8
    pi_e_agent = REINFORCEAgent(
        obs_dim=state_dim, 
        action_dim=action_dim, 
        action_scale=1.0
    )
    # Wrap for OPE
    pi_e_ope = OPEREINFORCEPolicy(pi_e_agent)
    
    # 4. Off-Policy Evaluation
    print("\n4. Running OPE Estimators...")
    
    # IPS
    ips_est, ips_diag = ips_estimate(logged_data, pi_e_ope, pi_behavior=pi_b_ope)
    print(f"IPS Estimate:   {ips_est:.4f} (ESS: {ips_diag['effective_sample_size']:.1f})")
    
    # SNIPS
    snips_est, snips_diag = snips_estimate(logged_data, pi_e_ope, pi_behavior=pi_b_ope)
    print(f"SNIPS Estimate: {snips_est:.4f}")
    
    # PDIS
    pdis_est, _ = pdis_estimate(logged_data, pi_e_ope, pi_behavior=pi_b_ope)
    print(f"PDIS Estimate:  {pdis_est:.4f}")
    
    # FQE (Model-based)
    print("\nFitting Q-function via FQE (100 iters)...")
    all_transitions = [t for traj in logged_data for t in traj.transitions]
    # Reduce dataset for speed in demo if needed, but 50 eps is small
    q_func, fqe_diag = fqe(
        all_transitions, 
        pi_e_ope, 
        num_iterations=100, 
        model_class="mlp", # Simple MLP for speed
        hidden_dims=[64, 64]
    )
    
    # Evaluate via FQE (average V(s_0) over start states)
    # Using the initial states from the logged data
    fqe_vals = []
    for traj in logged_data:
        s0 = traj.transitions[0].state
        fqe_vals.append(q_func.value(s0, pi_e_ope))
    fqe_est = np.mean(fqe_vals)
    print(f"FQE Estimate:   {fqe_est:.4f} (Loss: {fqe_diag['final_loss']:.4f})")
    
    # Doubly Robust
    dr_est, dr_diag = dr_estimate(logged_data, pi_e_ope, q_func, pi_behavior=pi_b_ope)
    print(f"DR Estimate:    {dr_est:.4f}")

    # 5. Ground Truth (On-Policy Monte Carlo)
    print("\n5. Computing Ground Truth (On-Policy)...")
    n_eval = 20
    returns = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            act = pi_e_ope.act(obs)
            obs, rew, done, truncated, _ = env.step(act)
            done = done or truncated
            ret += rew
        returns.append(ret)
    
    gt_return = np.mean(returns)
    print(f"Ground Truth:   {gt_return:.4f} (+/- {np.std(returns)/np.sqrt(n_eval):.2f})")
    
    print("\nSummary:")
    print(f"Behavior: {avg_return:.2f}")
    print(f"Target:   {gt_return:.2f}")
    print(f"IPS:      {ips_est:.2f}")
    print(f"DR:       {dr_est:.2f}")

if __name__ == "__main__":
    main()