import unittest
import numpy as np
import torch
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig

class TestREINFORCE(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 4
        self.action_dim = 2
        self.config = REINFORCEConfig(seed=42)
        self.agent = REINFORCEAgent(self.obs_dim, self.action_dim, config=self.config)

    def test_action_shape(self):
        obs = np.random.randn(self.obs_dim)
        action = self.agent.select_action(obs)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(isinstance(action, np.ndarray))

    def test_update_step(self):
        # Simulate short trajectory
        obs = np.random.randn(self.obs_dim)
        for _ in range(5):
            _ = self.agent.select_action(obs)
            self.agent.store_reward(1.0)
        
        # Check internal buffers before update
        self.assertEqual(len(self.agent.log_probs), 5)
        self.assertEqual(len(self.agent.rewards), 5)
        
        metrics = self.agent.update()
        
        # Check metrics keys
        self.assertIn("loss", metrics)
        self.assertIn("return_mean", metrics)
        self.assertIn("entropy", metrics)
        
        # Check buffers cleared
        self.assertEqual(len(self.agent.log_probs), 0)
        self.assertEqual(len(self.agent.rewards), 0)

    def test_entropy_gradients(self):
        # Verify that update changes parameters
        obs = np.random.randn(self.obs_dim)
        self.agent.select_action(obs)
        self.agent.store_reward(1.0)
        
        old_params = [p.clone() for p in self.agent.policy.parameters()]
        self.agent.update()
        
        # Check at least one param changed
        changed = False
        for p_old, p_new in zip(old_params, self.agent.policy.parameters()):
            if not torch.allclose(p_old, p_new):
                changed = True
                break
        self.assertTrue(changed, "Parameters should update after training step")

if __name__ == "__main__":
    unittest.main()
