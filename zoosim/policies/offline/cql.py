"""Conservative Q-Learning (CQL) for Continuous Actions.

Reference:
    Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning" (NeurIPS 2020).

This implementation adds the CQL regularization term to a standard Q-Network:
    L_CQL = alpha * (logsumexp(Q(s, a_rand)) - Q(s, a_data))
    
This pushes down Q-values for unseen actions, preventing overestimation on OOD data.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class CQLAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_max=5.0,
        lr=3e-4,
        alpha=1.0, # CQL weight
        device="cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.alpha = alpha
        self.device = device
        
        # Q-Network (State + Action -> Value)
        # We use 2 Q-networks for stability (Clipped Double Q)
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        
        self.optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=lr
        )

    def select_action(self, state):
        # For evaluation, we need to find argmax Q(s,a).
        # Since Q is a neural net, we use CEM (Cross-Entropy Method) or simple sampling.
        # Here we use simple random shooting for simplicity (or CEM if we had it imported).
        
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # Sample N random actions
        num_samples = 100
        actions = torch.FloatTensor(num_samples, self.action_dim).uniform_(-self.action_max, self.action_max).to(self.device)
        
        # Repeat state
        states = state_t.repeat(num_samples, 1)
        
        # Evaluate Q
        sa = torch.cat([states, actions], dim=1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        q = torch.min(q1, q2)
        
        best_idx = torch.argmax(q)
        return actions[best_idx].cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # 1. Standard Bellman Error
        # Since continuous, getting max_a Q(s', a') is hard.
        # In offline RL, we often just assume Q(s', a') ~ 0 for terminal or use a target policy.
        # For simplicity here, we assume single-step bandit (gamma=0) or just fit R.
        # If gamma > 0, we need an Actor to pick next action or use CEM.
        # Let's assume Bandit setting (gamma=0) for Search. Target is just Reward.
        
        target_q = rewards
        
        current_q1 = self.q1(torch.cat([states, actions], dim=1))
        current_q2 = self.q2(torch.cat([states, actions], dim=1))
        
        bellman_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 2. CQL Regularization
        # Minimize Q on random actions, Maximize Q on data actions
        
        # Generate random actions
        batch_size = states.size(0)
        random_actions = torch.FloatTensor(batch_size, 10, self.action_dim).uniform_(-self.action_max, self.action_max).to(self.device)
        states_exp = states.unsqueeze(1).repeat(1, 10, 1) # (B, 10, S)
        
        # Flatten
        states_flat = states_exp.reshape(-1, self.state_dim)
        actions_flat = random_actions.reshape(-1, self.action_dim)
        
        q1_rand = self.q1(torch.cat([states_flat, actions_flat], dim=1)).reshape(batch_size, 10, 1)
        q2_rand = self.q2(torch.cat([states_flat, actions_flat], dim=1)).reshape(batch_size, 10, 1)
        
        # LogSumExp over random actions
        cql_loss1 = torch.logsumexp(q1_rand, dim=1).mean() - current_q1.mean()
        cql_loss2 = torch.logsumexp(q2_rand, dim=1).mean() - current_q2.mean()
        
        total_loss = bellman_loss + self.alpha * (cql_loss1 + cql_loss2)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {"loss": total_loss.item(), "bellman": bellman_loss.item(), "cql": (cql_loss1+cql_loss2).item()}
