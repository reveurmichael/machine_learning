import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .model import ActorCritic
from torch.distributions import Categorical
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.clip_param = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def select_action(self, state, epsilon=0.0):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_probs, _ = self.actor_critic(state_tensor)
            
            # 使用epsilon-greedy策略
            if np.random.random() < epsilon:
                action = np.random.randint(0, action_probs.shape[1])
            else:
                action = torch.argmax(action_probs).item()
            
            return action

    def update(self, states, actions, rewards, next_states, dones):
        # 将数据转换为tensor
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 计算优势函数
        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
            advantages = rewards + (1 - dones) * 0.99 * next_values - values
            returns = advantages + values

        # 计算策略损失
        action_probs, value_preds = self.actor_critic(states)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        
        # PPO裁剪
        ratio = torch.exp(action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = F.mse_loss(value_preds, returns)
        
        # 熵正则化
        entropy = dist.entropy().mean()
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()