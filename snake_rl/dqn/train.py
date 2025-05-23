import os
import torch
import numpy as np
from ..snake_game import SnakeGame
from .agent import DQNAgent
from .replay_buffer import ReplayBuffer
from .utils import state_to_vector
import pygame
pygame.init()
pygame.font.init()

def main():
    # 创建模型保存目录
    os.makedirs('snakeq_rl/model/dqn', exist_ok=True)
    
    # 初始化环境和智能体
    env = SnakeGame()
    state_dim = 28  # 状态空间维度
    action_dim = 4  # 动作空间维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(state_dim, action_dim, device, lr=1e-4, gamma=0.99)
    buffer = ReplayBuffer(100000)  # 增加经验回放缓冲区大小
    
    # 训练参数
    max_episodes = 100000  # 增加到100,000代
    max_steps = 200  # 增加每轮最大步数
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999  # 减缓探索率衰减
    target_update = 10
    
    # 训练循环
    best_reward = float('-inf')
    episode_rewards = []
    total_steps = 0
    
    print("开始训练...")
    print(f"目标训练轮数: {max_episodes}")
    print(f"每轮最大步数: {max_steps}")
    print(f"初始探索率: {epsilon}")
    print(f"探索率衰减: {epsilon_decay}")
    print(f"最小探索率: {epsilon_min}")
    print(f"批量大小: {batch_size}")
    print(f"经验回放缓冲区容量: {buffer.capacity}")
    print(f"目标网络更新频率: {target_update}")
    print("=" * 50)
    
    # 设置时钟控制帧率
    clock = pygame.time.Clock()
    fps = 10  # 控制训练速度
    
    for episode in range(max_episodes):
        env.reset()
        state = state_to_vector(env)
        total_reward = 0
        steps = 0
        last_positions = []
        apples_eaten = 0
        
        while True:
            # 处理事件，允许用户退出
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
            
            # 选择动作
            action = agent.select_action(state, epsilon)
            move = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
            
            # 执行动作
            alive, apple_eaten = env.make_move(move)
            next_state = state_to_vector(env)
            
            # 获取当前头部位置
            head_pos = env.head_position
            apple_pos = env.apple_position
            
            # 计算奖励
            reward = 0
            if apple_eaten:
                reward = 30  # 吃到苹果的奖励
                apples_eaten += 1
                print(f"[Train] Apple eaten! Score: {env.score}, New apple at: {env.apple_position}")
            elif not alive:
                reward = -100  # 死亡的惩罚
            else:
                # 根据到苹果的距离给予奖励
                distance = np.linalg.norm(np.array(head_pos) - np.array(apple_pos))
                reward = -1  # 每步给予-1的惩罚
                
                # 惩罚重复位置
                if any(np.array_equal(head_pos, pos) for pos in last_positions[-3:]):
                    reward -= 1
            
            # 更新最近位置记录
            last_positions.append(head_pos.copy())  # 使用copy()避免引用问题
            if len(last_positions) > 5:
                last_positions.pop(0)
            
            # 存储经验
            buffer.push(state, action, reward, next_state, not alive)
            
            # 训练智能体
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = agent.update(batch)
                
                # 更新目标网络
                if total_steps % target_update == 0:
                    agent.update_target()
            
            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1
            
            # 更新显示
            env.draw()
            pygame.display.flip()
            
            # 控制帧率
            clock.tick(fps)
            
            # 检查是否结束
            if not alive or steps >= max_steps:
                break
        
        # 更新探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # 记录奖励
        episode_rewards.append(total_reward)
        
        # 打印训练进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_apples = np.mean([r/50 for r in episode_rewards[-100:] if r > 0])  # 估算平均吃到的苹果数
            print(f"Episode {episode + 1}/{max_episodes}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"平均吃到苹果数: {avg_apples:.2f}")
            print(f"当前探索率: {epsilon:.4f}")
            print(f"总步数: {total_steps}")
            print(f"经验回放缓冲区使用量: {len(buffer)}/{buffer.capacity}")
            print("-" * 30)
            
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save('snakeq_rl/model/dqn/dqn_best.pth')
                print(f"新的最佳模型已保存! 平均奖励: {best_reward:.2f}")
        
        # 定期保存模型
        if (episode + 1) % 1000 == 0:
            agent.save(f'snakeq_rl/model/dqn/dqn_episode_{episode + 1}.pth')
            print(f"模型已保存: dqn_episode_{episode + 1}.pth")

if __name__ == "__main__":
    main()