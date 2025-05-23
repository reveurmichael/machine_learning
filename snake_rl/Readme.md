# Snake Game Reinforcement Learning

这个项目实现了使用强化学习（PPO和DQN）来训练AI玩贪吃蛇游戏。

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- Pygame
- Streamlit

## 安装依赖

```bash
pip install torch numpy pygame streamlit
```

## 使用方法

### 图形界面（推荐）

使用 Streamlit 提供的图形界面，可以更方便地训练和测试模型：

```bash
streamlit run snakeq_rl/app.py
```

图形界面功能：
1. 训练页面：
   - 选择算法（PPO/DQN）
   - 调整训练参数
   - 实时显示训练进度
   - 自动保存模型

2. 测试页面：
   - 选择算法（PPO/DQN）
   - 选择要测试的模型文件
   - 实时显示测试过程

### 命令行方式

#### PPO (Proximal Policy Optimization)

#### 训练模型

```bash
python -m snakeq_rl.ppo.train
```

训练参数说明：
- 最大训练轮数：100000
- 每轮最大步数：200
- 初始探索率：1.0
- 探索率衰减：0.9999
- 最小探索率：0.01
- 批量大小：64
- PPO更新间隔：2048

训练过程中：
- 每100轮会显示训练进度，包括平均奖励、平均吃到苹果数、当前探索率等
- 当达到新的最佳平均奖励时，会自动保存最佳模型到 `snakeq_rl/model/ppo/ppo_best.pth`
- 每1000轮会保存一个检查点模型到 `snakeq_rl/model/ppo/ppo_episode_X.pth`

#### 测试模型

```bash
python -m snakeq_rl.ppo.test
```

默认会加载最佳模型 `snakeq_rl/model/ppo/ppo_best.pth`。测试过程中：
- 按 ESC 键可以退出测试
- 会显示当前得分和步数
- 游戏结束时会显示最终得分和总步数

### DQN (Deep Q-Network)

#### 训练模型

```bash
python -m snakeq_rl.dqn.train
```

训练参数说明：
- 最大训练轮数：100000
- 每轮最大步数：200
- 初始探索率：1.0
- 探索率衰减：0.9999
- 最小探索率：0.01
- 批量大小：64
- 目标网络更新频率：1000步

训练过程中：
- 每100轮会显示训练进度，包括平均奖励、平均吃到苹果数、当前探索率等
- 当达到新的最佳平均奖励时，会自动保存最佳模型到 `snakeq_rl/model/dqn/dqn_best.pth`
- 每1000轮会保存一个检查点模型到 `snakeq_rl/model/dqn/dqn_episode_X.pth`

#### 测试模型

```bash
python -m snakeq_rl.dqn.test
```

默认会加载最佳模型 `snakeq_rl/model/dqn/dqn_best.pth`。测试过程中：
- 按 ESC 键可以退出测试
- 会显示当前得分和步数
- 游戏结束时会显示最终得分和总步数

## 游戏规则

- 蛇可以通过方向键控制移动
- 吃到食物（红色方块）时，蛇身会变长，得分增加
- 撞到墙壁或自己的身体时游戏结束
- 目标是尽可能多地吃到食物，获得高分

## 奖励机制

- 吃到食物：+30分
- 撞墙或撞到自己：-100分
- 每步移动：-1分
- 重复位置：额外-1分

## 模型保存位置

- PPO模型：
  - 最佳模型：`snakeq_rl/model/ppo/ppo_best.pth`
  - 检查点：`snakeq_rl/model/ppo/ppo_episode_X.pth`

- DQN模型：
  - 最佳模型：`snakeq_rl/model/dqn/dqn_best.pth`
  - 检查点：`snakeq_rl/model/dqn/dqn_episode_X.pth`

## 注意事项

1. 训练过程中可以随时按 ESC 键退出
2. 训练时间可能较长，建议使用GPU进行训练
3. 如果训练效果不理想，可以尝试调整超参数
4. 测试时建议使用最佳模型，而不是检查点模型
5. 使用图形界面时，确保已安装 Streamlit
6. 图形界面支持实时显示训练和测试进度
