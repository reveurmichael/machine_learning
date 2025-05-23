import torch
import numpy as np
from .model import DQNNet

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-3, gamma=0.99):
        self.device = device
        self.gamma = gamma
        self.policy_net = DQNNet(state_dim, action_dim).to(device)
        self.target_net = DQNNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, 4)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self, batch):
        states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.functional.mse_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        """保存模型到指定路径"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
        
    def load(self, path):
        """从指定路径加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']