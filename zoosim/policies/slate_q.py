"""SlateQ Agent (Simplified) for Deep Learning to Rank.

Implements a Deep Q-Network that scores (Context, Item) pairs.
Used for constructing rankings by sorting Q-values.

Reference:
    Ie et al. "SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets" (IJCAI 2019).
    (We implement a simplified single-step version suitable for the bandit setting).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class SlateQNetwork(nn.Module):
    def __init__(self, context_dim, item_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = context_dim + item_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, context, item_features):
        """
        Args:
            context: (Batch, ContextDim)
            item_features: (Batch, NumItems, ItemDim)
        Returns:
            scores: (Batch, NumItems)
        """
        # Expand context to match items
        # context: (B, C) -> (B, 1, C) -> (B, N, C)
        batch_size, num_items, _ = item_features.shape
        context_expanded = context.unsqueeze(1).expand(-1, num_items, -1)
        
        # Concat
        x = torch.cat([context_expanded, item_features], dim=-1)
        
        # Forward (B, N, C+I) -> (B, N, 1)
        q_values = self.net(x).squeeze(-1)
        return q_values

class SlateQAgent:
    def __init__(
        self, 
        context_dim, 
        item_dim, 
        lr=1e-3, 
        gamma=0.99, 
        batch_size=32, 
        buffer_size=10000,
        device="cpu"
    ):
        self.context_dim = context_dim
        self.item_dim = item_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        
        self.q_net = SlateQNetwork(context_dim, item_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.replay_buffer = deque(maxlen=buffer_size)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, obs):
        """
        Returns scores for all candidates.
        obs['context']: (C,)
        obs['candidate_features']: (N, I)
        """
        context = torch.FloatTensor(obs['context']).unsqueeze(0).to(self.device)
        candidates = torch.FloatTensor(obs['candidate_features']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(context, candidates).squeeze(0).cpu().numpy()
            
        # Epsilon-Greedy exploration
        # For ranking, this usually means shuffling the top-k or adding noise.
        # We will simply add noise to the scores.
        if np.random.rand() < self.epsilon:
            noise = np.random.normal(0, 1.0, size=q_values.shape)
            return q_values + noise
        
        return q_values

    def store_transition(self, context, candidates, action_scores, reward):
        """
        context: np array
        candidates: np array
        action_scores: np array (scores used to rank)
        reward: float
        """
        self.replay_buffer.append((context, candidates, action_scores, reward))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
            
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        contexts = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        candidates = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        
        # In a true SlateQ, we would compute Target = r + gamma * Q(s', a').
        # Here, we treat it as a contextual bandit (gamma=0 effectively for this demo),
        # trying to predict the slate reward.
        
        # We assume Q_slate approx Sum(Q_item) for the top-k items.
        # But wait, 'candidates' contains ALL candidates, not just the slate.
        # We need to know which items were actually SHOWN (the slate).
        # In our env, the top-K items by 'action_scores' are shown.
        
        # We need to re-compute the slate indices based on the stored action_scores
        # This is computationally expensive if we do it here.
        # Ideally, we should store the 'slate_indices' in the buffer.
        
        # Simplified Training Objective:
        # We treat the PROBLEM as "Learning to Rank". 
        # We can use a Pairwise Loss or Pointwise MSE against a heuristic target.
        # Since we only have scalar slate reward, attributing it is hard without the PBM decomposition.
        
        # Let's fallback to the simplest "Deep RL" approach that works:
        # Treat the "Context + Best Item" as a sample with reward.
        # Or, use the REINFORCE logic but with a Value Baseline.
        
        # Actually, let's implement the LTV (Learned Value) approach:
        # Q(s, i) -> predicted probability of conversion * value.
        # We minimize MSE( Sum(Q(i) for i in Slate) - Reward ).
        # This encourages the Q-values to sum up to the observed reward.
        
        # 1. Forward pass
        q_all = self.q_net(contexts, candidates) # (B, N)
        
        # 2. Identify Slate Items (Top K)
        # For this demo, let's assume K=5 (Env config default usually)
        K = 5
        # We need to sort the q_all to find which ones WOULD be picked by the current policy?
        # No, we need the items that WERE picked in the transition.
        # We need to pass 'action_scores' to 'update' to reconstruct the slate.
        
        stored_scores = np.array([b[2] for b in batch])
        # We find the indices of the top-K scores for each batch element
        top_k_indices = np.argsort(-stored_scores, axis=1)[:, :K]
        
        # Gather Q-values for the slate items
        # top_k_indices: (B, K)
        # q_all: (B, N)
        
        slate_q_values = torch.gather(q_all, 1, torch.tensor(top_k_indices).to(self.device)) # (B, K)
        
        # Predicted Slate Value = Sum(Q_item)
        # This assumes items are independent additives.
        predicted_rewards = slate_q_values.sum(dim=1)
        
        loss = nn.functional.mse_loss(predicted_rewards, rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {"loss": loss.item(), "epsilon": self.epsilon}

