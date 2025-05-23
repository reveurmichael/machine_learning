import numpy as np

def state_to_vector(game):
    """
    返回28维状态向量，与SnakeQ一致：
    - 前4个：到墙的距离 (0-1)
    - 中间8个：是否看到苹果 (0/1)
    - 接下来8个：到蛇身的距离 (0-1)
    - 最后8个：头部和尾部方向 (one-hot)
    """
    N = game.grid_size
    head_pos = game.head_position
    head_dir = game.current_direction
    tail_dir = game.snake_positions[-2] - game.snake_positions[-1] if len(game.snake_positions) > 1 else head_dir
    
    # 1. 到墙的距离 (4维)
    wall_dist = np.array([
        head_pos[1] / (N-1),  # 上
        (N-1 - head_pos[0]) / (N-1),  # 右
        (N-1 - head_pos[1]) / (N-1),  # 下
        head_pos[0] / (N-1)  # 左
    ])
    
    # 2. 是否看到苹果 (8维)
    apple_pos = game.apple_position
    see_apple = np.zeros(8)
    # 检查8个方向是否能看到苹果
    directions = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]
    for i, (dx, dy) in enumerate(directions):
        x, y = head_pos
        while 0 <= x < N and 0 <= y < N:
            if np.array_equal([x, y], apple_pos):
                see_apple[i] = 1
                break
            x += dx
            y += dy
    
    # 3. 到蛇身的距离 (8维)
    body_dist = np.zeros(8)
    for i, (dx, dy) in enumerate(directions):
        x, y = head_pos
        dist = 0
        while 0 <= x < N and 0 <= y < N:
            if any(np.array_equal([x, y], pos) for pos in game.snake_positions[1:]):
                body_dist[i] = 1 - dist / (N-1)
                break
            x += dx
            y += dy
            dist += 1
    
    # 4. 头部和尾部方向 (8维)
    dir_to_onehot = {
        (0, -1): [1, 0, 0, 0],  # 上
        (1, 0): [0, 1, 0, 0],   # 右
        (0, 1): [0, 0, 1, 0],   # 下
        (-1, 0): [0, 0, 0, 1]   # 左
    }
    
    # 处理head_dir为None的情况
    if head_dir is None:
        head_dir_onehot = [0, 0, 0, 0]
    else:
        head_dir_onehot = dir_to_onehot.get(tuple(head_dir), [0, 0, 0, 0])
    
    # 处理tail_dir为None的情况
    if tail_dir is None:
        tail_dir_onehot = [0, 0, 0, 0]
    else:
        tail_dir_onehot = dir_to_onehot.get(tuple(tail_dir), [0, 0, 0, 0])
    
    # 组合所有状态
    state = np.concatenate([
        wall_dist,
        see_apple,
        body_dist,
        head_dir_onehot,
        tail_dir_onehot
    ])
    
    return state