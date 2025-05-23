import pygame
import torch
from ..snake_game import SnakeGame
from .agent import PPOAgent  # 或 DQNAgent
from .utils import state_to_vector
import time

pygame.init()
pygame.font.init()

def test_model(model_path='snakeq_rl/model/ppo/ppo_best.pth'):  # 或 dqn_best.pth
    # 初始化环境和智能体
    env = SnakeGame()
    state_dim = 28
    action_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建智能体并加载模型
    agent = PPOAgent(state_dim, action_dim, device)  # 或 DQNAgent
    agent.load(model_path)
    
    # 初始化pygame
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("PPO Snake - Testing")  # 或 "DQN Snake - Testing"
    
    # 设置时钟控制帧率
    clock = pygame.time.Clock()
    fps = 10  # 控制显示速度
    
    # 测试循环
    running = True
    env.reset()
    state = state_to_vector(env)
    total_reward = 0
    steps = 0
    last_score_time = time.time()  # 记录最后得分时间
    last_score = 0  # 记录上一次的得分
    
    print("开始测试模型...")
    print(f"使用模型: {model_path}")
    print("按ESC退出")
    print("=" * 50)
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 选择动作（epsilon=0，纯利用）
        action = agent.select_action(state, epsilon=0.0)
        move = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
        
        # 执行动作
        alive, apple_eaten = env.make_move(move)
        next_state = state_to_vector(env)
        
        # 更新状态
        state = next_state
        steps += 1
        
        # 显示游戏状态
        env.draw()
        pygame.display.flip()
        
        # 控制帧率
        clock.tick(fps)
        
        # 检查是否需要重新开始
        if not alive or (time.time() - last_score_time > 30 and env.score == last_score):
            print(f"游戏结束! 得分: {env.score}, 步数: {steps}")
            print("重新开始实验...")
            env.reset()
            state = state_to_vector(env)
            total_reward = 0
            steps = 0
            last_score_time = time.time()
            last_score = 0
            continue
        
        # 更新得分记录
        if env.score > last_score:
            last_score = env.score
            last_score_time = time.time()
    
    pygame.quit()

if __name__ == "__main__":
    test_model()