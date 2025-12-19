"""Experiment 10.1: Drift Detection and Guardrails Demo.

Simulates a non-stationary environment where user preferences shift abruptly.
Demonstrates:
1. LinUCB failure (lag/noise) under drift.
2. SafetyMonitor detecting reward degradation.
3. Fallback to a safe static policy.
4. Automatic recovery via periodic probing (managed by SafetyMonitor).
5. Comprehensive monitoring: Reward, GMV, Stability (Delta-Rank), and Latency.

Evaluation Metrics Monitored:
- Reward (Composite)
- GMV (Gross Merchandise Value)
- Stability (Delta-Rank@k) [Syllabus Requirement]
- System Health (Latency) [Syllabus Requirement]
- Recovery Time (N episodes)

References:
    - Chapter 10: Robustness to Drift and Guardrails
    - [ALG-10.1] Drift Detection Loop
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

from zoosim.monitoring.drift import DriftDetector, PageHinkley, PageHinkleyConfig
from zoosim.monitoring.guardrails import GuardrailConfig, SafetyMonitor
from zoosim.monitoring.metrics import compute_delta_rank_at_k, compute_gmv
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.templates import BoostTemplate

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 12)


@dataclass
class SimulationMetrics:
    rewards: List[float] = field(default_factory=list)
    gmv: List[float] = field(default_factory=list)
    cm2: List[float] = field(default_factory=list)      # New: Contribution Margin 2
    clicks: List[int] = field(default_factory=list)     # New: Click events (0/1)
    conversions: List[int] = field(default_factory=list) # Renamed from cvr for clarity
    actions: List[int] = field(default_factory=list)
    modes: List[int] = field(default_factory=list)  # 0: Primary, 1: Fallback
    stability: List[float] = field(default_factory=list) # Delta-Rank@k
    latency: List[float] = field(default_factory=list)   # ms


@dataclass
class DriftScenarioConfig:
    """Configuration for drift scenario parameters."""
    # Phase 1 parameters
    phase1_ctr: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.20, 0.15, 0.12]))
    phase1_cvr_given_click: np.ndarray = field(default_factory=lambda: np.array([0.50, 0.60, 0.53, 0.50]))

    # Phase 2 parameters (Drift)
    phase2_ctr: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.05, 0.15, 0.25]))
    phase2_cvr_given_click: np.ndarray = field(default_factory=lambda: np.array([0.50, 0.40, 0.53, 0.60]))

    # Static properties
    prices: np.ndarray = field(default_factory=lambda: np.array([50.0, 60.0, 45.0, 55.0]))


class StaticPolicy:
    """Safe fallback policy that always selects a fixed template."""
    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action

    def select_action(self, features: np.ndarray) -> int:
        return self.fixed_action

    def update(self, action: int, features: np.ndarray, reward: float) -> None:
        pass  # Static policy does not learn


class DriftEnvironment:
    """Simulates a non-stationary e-commerce environment."""
    
    def __init__(self, n_templates: int, drift_step: int, config: DriftScenarioConfig = None):
        self.n_templates = n_templates
        self.drift_step = drift_step
        self.config = config or DriftScenarioConfig()
        self.t = 0
        
        # Define Ground Truths
        # Funnel: Impression -> Click -> Conversion
        # Phase 1: T1 (High Margin) is best (Good CTR * Good CVR)
        # Phase 2: T3 (Popular) becomes best (High CTR * High CVR), T1 crashes
        
        # Resulting Prob(Conv): ~[0.05, 0.12, 0.08, 0.06] in Phase 1
        # Resulting Prob(Conv): ~[0.05, 0.02, 0.08, 0.15] in Phase 2

        # Base Ranking (Item IDs) for Stability Check
        self.base_ranking = list(range(20))

    def step(self, action: int) -> Tuple[float, Dict[str, Any]]:
        """Simulate one step: Action -> Click -> Conversion -> Reward."""
        self.t += 1
        
        # Determine current regime
        if self.t < self.drift_step:
            ctr_means = self.config.phase1_ctr
            cvr_cond_means = self.config.phase1_cvr_given_click
        else:
            ctr_means = self.config.phase2_ctr
            cvr_cond_means = self.config.phase2_cvr_given_click
            
        # 1. Simulate Click (Bernoulli)
        true_ctr = ctr_means[action] if action < len(ctr_means) else 0.0
        click = np.random.binomial(1, true_ctr)

        # 2. Simulate Conversion (Bernoulli)
        if click:
            true_cvr = cvr_cond_means[action] if action < len(cvr_cond_means) else 0.0
            conversion = np.random.binomial(1, true_cvr)
        else:
            conversion = 0
        
        # 3. Calculate Business Metrics
        # We treat the action as selecting a "winner" item type for simplicity in GMV calc
        gmv = compute_gmv(np.array([self.config.prices[action]]), np.array([conversion]))
        
        # CM2 (Margin 2) - assume ~40% margin for demo
        # NOTE: This is a simplification ("approx_cm2"). Real CM2 = GMV - COGS - Logistics - Marketing.
        # See Exercise 10.3 for per-template margin implementation.
        approx_cm2 = gmv * 0.4 
        
        # 4. Composite Reward
        # R = w1*GMV + w2*Click (simplified here to just GMV/Conversion driven)
        reward = approx_cm2 + np.random.normal(0, 0.5)
        
        # 5. Simulate Ranking for Delta-Rank@k
        # Different actions produce slightly different perturbations of the base ranking
        current_ranking = self.base_ranking.copy()
        if action == 1:
            current_ranking[:5] = reversed(current_ranking[:5])
        elif action == 2:
            sub = current_ranking[:10]
            np.random.shuffle(sub)
            current_ranking[:10] = sub
        elif action == 3:
            current_ranking = current_ranking[2:] + current_ranking[:2]
            
        return reward, {
            "gmv": gmv,
            "cm2": approx_cm2,
            "click": click,
            "conversion": conversion,
            "ranking": current_ranking
        }


def get_dummy_templates() -> List[BoostTemplate]:
    """Create standard templates for simulation."""
    return [
        BoostTemplate(id=0, name="T0_Neutral", description="Baseline", boost_fn=lambda x: 0.0),
        BoostTemplate(id=1, name="T1_HighMargin", description="Margin Boost", boost_fn=lambda x: 0.0),
        BoostTemplate(id=2, name="T2_Discount", description="Discount Boost", boost_fn=lambda x: 0.0),
        BoostTemplate(id=3, name="T3_Popular", description="Popularity Boost", boost_fn=lambda x: 0.0),
    ]


def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def calculate_recovery_time(
    rewards: List[float], drift_step: int, baseline_window: int = 200, recovery_threshold: float = 0.9
) -> int:
    """Calculate episodes needed to recover to pre-drift baseline performance."""
    if len(rewards) < drift_step + baseline_window:
        return -1
        
    # Baseline: Average reward in window before drift
    baseline = np.mean(rewards[drift_step - baseline_window : drift_step])
    
    # Check post-drift
    post_drift = rewards[drift_step:]
    window = 50
    for i in range(len(post_drift) - window):
        current_perf = np.mean(post_drift[i : i + window])
        # Handle case where baseline is negative or zero appropriately if needed
        if baseline > 0.1 and current_perf >= baseline * recovery_threshold:
            return i
        elif baseline <= 0.1 and current_perf > baseline: # Relaxed check for low baseline
             return i
            
    return -1


def simulate_drift_scenario(n_steps: int = 3000):
    """Run simulation satisfying Chapter 10 requirements."""
    
    drift_step = 1500
    templates = get_dummy_templates()
    feature_dim = 5
    
    # 1. Policies
    # Primary: LinUCB
    primary_config = LinUCBConfig(alpha=1.5, lambda_reg=1.0)
    primary_policy = LinUCB(templates, feature_dim, config=primary_config)
    
    # Safe: Static Policy (Neutral T0)
    safe_policy = StaticPolicy(fixed_action=0)
    
    # 2. Guardrails & Monitor
    ph_config = PageHinkleyConfig(
        threshold=30.0,      # Sensitivity threshold
        delta=0.01,          # Tolerable drift magnitude
        min_instances=30
    )
    drift_detector = PageHinkley(ph_config)
    
    guard_config = GuardrailConfig(
        enable_drift_detection=True,
        drift_patience=5,
        probe_frequency=100, # Automatic recovery attempt every 100 episodes
        enable_stability_check=True
    )
    
    monitor = SafetyMonitor(
        primary_policy=primary_policy,
        safe_policy=safe_policy,
        config=guard_config,
        drift_detector=drift_detector
    )
    
    # 3. Environment
    env = DriftEnvironment(len(templates), drift_step)
    
    # Data logging
    sim_data = SimulationMetrics()
    prev_ranking = list(range(20)) # Init
    
    print(f"Starting Simulation (Steps={n_steps}, Drift@{drift_step})...")
    print("Phase 1: T1 (High Margin) is optimal.")
    
    for t in range(n_steps):
        # Simulate Latency (System Health Metric)
        # Normal: ~20ms, Spike: ~100ms (rare)
        start_time = time.perf_counter()
        
        # Context
        features = np.random.randn(feature_dim)
        features /= np.linalg.norm(features)
        
        # Action
        action = monitor.select_action(features)
        
        # Step
        reward, info = env.step(action)
        
        # Latency Calc
        # NOTE: This is a synthetic metric for demo purposes to demonstrate monitoring.
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        # Add synthetic noise/spikes to latency for realism
        latency_ms += np.random.gamma(shape=2.0, scale=10.0) 
        
        # Update
        monitor.update(action, features, reward)
        
        # Metrics Calculation
        curr_ranking = info['ranking']
        delta_rank = compute_delta_rank_at_k(prev_ranking, curr_ranking, k=10)
        prev_ranking = curr_ranking
        
        # Log
        sim_data.rewards.append(reward)
        sim_data.gmv.append(info['gmv'])
        sim_data.cm2.append(info['cm2'])
        sim_data.clicks.append(info['click'])
        sim_data.conversions.append(info['conversion'])
        sim_data.actions.append(action)
        sim_data.modes.append(1 if monitor.in_fallback_mode else 0)
        sim_data.stability.append(delta_rank)
        sim_data.latency.append(latency_ms)
        
        if t == drift_step:
             print("⚠️  PHASE 2 START: Preference Drift Occurred! (T1 crashes, T3 optimal)")

    # 4. Analysis
    recovery_time = calculate_recovery_time(sim_data.rewards, drift_step)
    print(f"\nSimulation Complete.")
    print(f"Time to Recover (90% of baseline): {recovery_time if recovery_time > 0 else '> End'} episodes")
    
    plot_results(sim_data, drift_step)


def plot_results(data: SimulationMetrics, drift_step: int):
    t = np.arange(len(data.rewards))
    
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 20))
    
    # Plot 1: Reward, CM2 & Fallback
    ax1 = axes[0]
    ma_rewards = moving_average(data.rewards, n=50)
    ma_cm2 = moving_average(data.cm2, n=50)
    
    ax1.plot(t[len(t)-len(ma_rewards):], ma_rewards, label="Avg Reward", color="blue", linewidth=2)
    ax1.plot(t[len(t)-len(ma_cm2):], ma_cm2, label="Avg CM2", color="green", linewidth=1.5, alpha=0.8)
    ax1.axvline(x=drift_step, color="red", linestyle="--", label="Drift Event")
    
    # Highlight Fallback
    fallback_indices = np.where(np.array(data.modes) == 1)[0]
    if len(fallback_indices) > 0:
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.fill_between(t, 0, 1, where=(np.array(data.modes)==1), facecolor='orange', alpha=0.3, transform=trans, label="Fallback Mode")

    ax1.set_ylabel("Value")
    ax1.set_title("Financial Metrics: Reward & CM2")
    ax1.legend(loc="upper left")

    # Plot 2: Engagement (CTR & CVR) - New Syllabus Req
    ax2 = axes[1]
    ma_ctr = moving_average(data.clicks, n=50)
    ma_cvr = moving_average(data.conversions, n=50)
    
    ax2.plot(t[len(t)-len(ma_ctr):], ma_ctr, label="CTR (Clicks)", color="magenta")
    ax2.plot(t[len(t)-len(ma_cvr):], ma_cvr, label="CVR (Conversions)", color="cyan")
    ax2.axvline(x=drift_step, color="red", linestyle="--")
    ax2.set_ylabel("Rate [0,1]")
    ax2.set_title("Engagement Metrics: CTR & CVR")
    ax2.legend(loc="upper left")

    # Plot 3: Action Distribution (Policy Adaptation)
    ax3 = axes[2]
    ax3.scatter(t, data.actions, s=2, alpha=0.3, c=data.actions, cmap="viridis", label="Action")
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(["T0 (Safe)", "T1 (Margin)", "T2 (Disc)", "T3 (Pop)"])
    ax3.set_ylabel("Template")
    ax3.axvline(x=drift_step, color="red", linestyle="--")
    ax3.set_title("Policy Selection")
    
    # Plot 4: Stability (Delta-Rank)
    ax4 = axes[3]
    ma_stab = moving_average(data.stability, n=50)
    ax4.plot(t[len(t)-len(ma_stab):], ma_stab, label="Delta-Rank@10 (Churn)", color="purple")
    ax4.axvline(x=drift_step, color="red", linestyle="--")
    ax4.set_ylabel("Churn [0,1]")
    ax4.set_title("Stability Metric (Delta-Rank@10)")
    
    # Plot 5: Latency
    ax5 = axes[4]
    # Downsample for clearer plot if needed, or moving average
    ma_lat = moving_average(data.latency, n=50)
    ax5.plot(t[len(t)-len(ma_lat):], ma_lat, label="Latency (ms)", color="gray")
    ax5.axvline(x=drift_step, color="red", linestyle="--")
    ax5.set_ylabel("Latency (ms)")
    ax5.set_xlabel("Episode")
    ax5.set_title("System Health: Latency")

    plt.tight_layout()
    output_path = "ch10_drift_demo.png"
    plt.savefig(output_path)
    print(f"Plot saved to '{output_path}'.")


if __name__ == "__main__":
    simulate_drift_scenario()