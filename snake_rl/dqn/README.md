# DQN 贪吃蛇训练 (PyTorch)

## 目录结构

- model.py: Q 网络结构
- agent.py: DQN 智能体
- replay_buffer.py: 经验回放
- utils.py: 状态向量编码
- train.py: 训练主循环

## 依赖

- numpy
- torch

## 运行方法

```bash
cd snakeq_pytorch
python -m dqn.train
```

## 说明
- 状态向量为棋盘 one-hot 展开 (N*N*3)
- 动作空间为 [UP, DOWN, LEFT, RIGHT]
- 训练过程中自动保存模型参数
- 兼容 snakeq_pytorch 的 SnakeGame 环境 