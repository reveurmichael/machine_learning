
# Chinse Version 中文版本任务描述

## 任务

Task 1 和 Task 2 任选一个。

### Task 1

做一个类似的，推箱子游戏 Sokoban Game。使用如下的方法玩游戏。

### Task 2

做一个类似的，俄罗斯方块游戏。Tetris Game。使用如下的方法玩游戏。


## 方法与比较


### 启发式 + 机器学习

通过 heuristic 生成各种好的步骤、数据。然后机器学习来进行**分类**(或者回归)学习。

| 比较维度                  | Sokoban（推箱子）             | Tetris（俄罗斯方块）            | Snake（贪吃蛇）            |
| --------------------- | ------------------------ | ------------------------ | --------------------- |
| **启发式设计难度**           | ⭐⭐⭐⭐ 高（避免死局、路径规划复杂）      | ⭐⭐ 中（局部贪婪策略如“消最多行”相对容易）  | ⭐⭐ 中（避墙 + 吃最近食物较易设计）  |
| **数据生成的自动化程度**        | ⭐⭐⭐ 较难（依赖强规划器，如 BFS/A\*） | ⭐⭐⭐⭐ 高（轻松生成局部最优数据）       | ⭐⭐⭐⭐ 高（可批量生成状态-动作对）   |
| **数据标签清晰性**           | ⭐⭐⭐⭐ 清晰（动作唯一性强）          | ⭐⭐ 一般（动作选择可多样）           | ⭐⭐⭐ 清晰（吃/不吃，转向或直行）    |
| **监督学习适配性**           | ⭐⭐⭐ 适配性中等（状态编码复杂）        | ⭐⭐⭐⭐ 高（特征简单，如堆叠形状 + 当前块） | ⭐⭐⭐⭐ 高（可直接用CNN或MLP建模） |
| **适合分类还是回归建模**        | 分类（动作：上下左右），可配合路径评分回归    | 分类（左、右、旋转、掉落），可预测预期得分    | 分类（转向动作）+ 回归（食物距离）    |
| **泛化能力**              | ⭐⭐ 一般（关卡依赖强）             | ⭐⭐⭐⭐ 较强（堆叠模式泛化好）         | ⭐⭐⭐ 一般（局部策略泛化，但路径规划差） |
| **可作为 RL warm-start** | ✅ 是（提供初始策略指导）            | ✅ 是（可用于初始化行为策略）          | ✅ 是（提供导航策略或避障经验）      |
| **是否适合教学入门**          | ⚠️ 难度略高（需讲解状态空间编码）       | ✅ 是（规则清晰，适合讲局部最优策略学习）    | ✅ 是（结构简单，适合入门教学和调试）   |


🚀 拓展：也可以配合 LLM
你甚至可以让 LLM 来生成启发式策略样本，再训练另一个小模型去模仿它。这种思路被称为：
- Distillation of LLM policy
- LLM-as-teacher

### 强化学习

| 游戏类型          | Sokoban（推箱子）                        | Tetris（俄罗斯方块）                              | Snake（贪吃蛇）                         |
| ------------- | -------------------------- | -------------------------- | -------------------------- |
| **环境类型**      | 离散 + 回合制                            | 离散 + 回合制（去除重力与时间概念）                        | 离散 + 回合制                           |
| **训练难度**      | ⭐⭐⭐⭐（非常高）                           | ⭐⭐⭐（中等偏高）                                  | ⭐⭐（中等）                             |
| **收敛速度**      | 慢（稀疏奖励）                             | 中等（逐步获得奖励）                                 | 快（每吃一次奖励）                          |
| **奖励稀疏性**     | 极高（必须完成目标才能奖励）                      | 中等（每行消除有奖励，但回合制下更慢）                        | 低（吃到食物立即奖励）                        |
| **长期依赖性**     | 强（一步错误导致不可逆局面）                      | 中（局部最优可能影响后期，但影响较慢）                        | 强（路径选错可能走入死局）                      |
| **适合的 RL 算法** | PPO, A3C, MCTS, Curriculum Learning | DQN, PPO, Evolutionary, Imitation Learning | DQN, PPO, A2C, Curriculum Learning |
| **状态空间大小**    | 超大（墙体 + 箱子 + 玩家位置组合）                | 巨大（堆叠结构 + 当前/下一个方块状态）                      | 中等偏大（蛇身 + 食物 + 障碍物 + 方向）           |
| **动作空间复杂性**   | 小（上下左右 + 推）                         | 中（左右移动 + 旋转 + 放置）                          | 小（上下左右移动）                          |



### Curriculum Learning

**Main idea for Snake Game:** Progressively increase the complexity of the Snake Game environment (e.g., smaller grid → larger grid, no walls → walls, slow speed → faster). Use curriculum learning techniques to gradually train RL agents to handle more complex scenarios.

- https://web.stanford.edu/class/aa228/reports/2020/final16.pdf
- https://github.com/greentfrapp/snake



| 对比维度                       | Sokoban                     | Tetris            | Snake               |
| -------------------------- | -------------------------- | ----------------- | ------------------- |
| **状态空间增长方式**               | 可控：通过地图复杂度调整（箱子数）           | 不可控：输入随机、状态动态生成   | 可控：食物/地图大小调整        |
| **任务分解性（可逐步加难）**           | ✅ 清晰：1箱 → 2箱 → 多箱           | ❌ 难分解：消几行不是分阶段的目标 | ✅ 明确：短蛇 → 长蛇 → 复杂障碍 |
| **任务目标清晰性**                | ✅ 终态明确（所有箱子到达目标）            | ⚠️ 模糊目标（分数最大化、无限） | ✅ 清晰目标（吃越多越好）       |
| **局部最优陷阱**                 | ✅ 严重（死局多）                   | ⚠️ 可控（堆错影响未来）     | ✅ 有（撞自己/空间封死）       |
| **reward 稀疏性**             | ✅ 极高（终点才给分）                 | ❌ 密集（每行消除得分）      | ⚠️ 适中（吃食物给分）        |
| **Curriculum 可设计方式**       | ✅ 地图分层 + 箱子数量 + 步数限制        | ⚠️ 难设计（无法阶段化目标）   | ✅ 地图尺寸、障碍数量、蛇身初始长度  |
| **对探索策略的需求**               | ✅ 高（需避免不可逆陷阱）               | ⚠️ 中（高分策略难以学到）    | ✅ 高（探索路径避免撞）        |
| **LLM 融入 Curriculum 的可行性** | ✅ 高：辅助 map 分层设计、deadlock 检测 | ⚠️ 难：辅助设计策略很抽象    | ✅ 可辅助：描述局势或预测食物策略   |


### LLM 规划
| **比较维度**          | **Sokoban（推箱子）**                | **Tetris（俄罗斯方块）**            | **Snake（贪吃蛇）**                     |
| ----------------- | -------------------------- | -------------------------- | -------------------------- |
| **输入状态复杂度**       | 中等：地图 + 玩家 + 箱子 + 目标点           | 高：堆叠形状 + 当前方块 + 下一个方块 + 可能落点 | 中等偏低：蛇身坐标序列 + 食物位置 + 障碍或边界位置       |
| **目标清晰度**         | 明确：所有箱子推到目标点                    | 模糊：最大化分数，策略不是唯一              | 明确：吃最多食物、不撞墙或自身                    |
| **动作结构清晰性**       | 高：动作离散，语义明确（上下左右 + 推）           | 中：旋转 + 左右移动 + 落下，顺序依赖较强      | 高：离散动作（上下左右），每步语义明确                |
| **动作序列的规划深度**     | 高：需避免陷阱，推错即无解                   | 中：一块一策略，局部最优可接受              | 中：需合理避障 + 长期预判，尤其蛇身较长时             |
| **语言描述的可表达性**     | 高：自然语言可描述状态 + 步骤（例如“将左侧箱子推向角落”） | 中：难以准确描述堆叠后的未来结构             | 高：描述“蛇应向上走两步以吃到食物并避免回头”等策略直观自然语言表达 |
| **LLM 生成动作的可解释性** | 强：一步一步验证是否推动正确，路径显式             | 弱：难从语言直接推理高分策略，易陷入次优积木结构     | 中等：每步是否撞墙/吃食物易验证，但长期路径选择需环境模拟      |
| **可逆性 / 可验证性**    | 高：每步可验证是否合法 / 步数较少              | 中：有些状态积木结构复杂，无法完全逆推          | 中：早期状态可逆，蛇变长后路径验证需模拟支持             |
| **状态 → 计划映射复杂度**  | ⭐⭐⭐⭐（高：组合爆炸 + 死局多）              | ⭐⭐（中：贪心策略较好，但全局最优复杂）         | ⭐⭐⭐（中高：尤其后期蛇身长，路径密集）               |


### LLM Eureka 奖励函数设计

| 对比维度                     | Sokoban（推箱子）         | Tetris（俄罗斯方块）              | Snake（贪吃蛇）              |
| ------------------------ | -------------------- | -------------------------- | ----------------------- |
| **目标清晰**                 | ✅ 明确：所有箱子入洞          | ⚠️ 相对模糊：最大化得分或清行           | ✅ 明确：吃到食物，避免撞墙或咬到自己     |
| **LLM 判断奖励时是否容易**        | ✅ 容易判断箱子是否到位、是否卡死    | ⚠️ 难以判断堆叠质量、未来潜在行清除        | ✅ 容易判断是否吃到食物和是否碰撞       |
| **reward 是否稀疏**          | ✅ 很稀疏（更需LLM来补充）      | ❌ 较密集（系统本身已有 clear reward） | ⚠️ 中等（吃食物奖励明确，但存活时间无奖励） |
| **LLM 奖励设计的增益空间**        | ✅ 高（帮助解决稀疏/陷阱状态）     | ⚠️ 中（策略差异不一定立刻体现）          | ✅ 高（可设计更细腻的生存与路径奖励）     |
| **解释性（Explainability）**  | ✅ 强，可逐步分析箱子状态        | ⚠️ 弱，最终得分与中间策略相关性低         | ✅ 中，可解释每步吃食与避障行为        |
| **人类偏好是否能被 LLM 模拟**      | ✅ 高，如“不要堵住角落”“步骤尽量少” | ⚠️ 模糊，如“堆得平稳”“为下一块留空间”     | ✅ 高，如“优先吃近食物”“避免死路”     |
| **LLM reward 对策略变化是否敏感** | ✅ 高（轻微错误就陷入死局）       | ⚠️ 中（策略变化未必立即体现好坏）         | ✅ 高（小动作差异会直接影响生存和得分）    |


## 其他方法

## Imitation Learning with Human Gameplay
Have students play the Snake Game and collect gameplay trajectories. Then, train agents using Behavioral Cloning or Inverse Reinforcement Learning to mimic human strategies.


### SnakeGTP

Fine tune LLM models for playing Snake Game.

### Visual Reasoning using Vision LLM 

### MoE-Based Strategy Selector
Use a Mixture of Experts (MoE) setup where different LLMs (Mistral 7B, DeepSeek, etc.) represent different “strategic minds” (e.g., aggressive, conservative). Train a controller model to switch among them dynamically based on game state.

### RL Safety via LLM Filtering
Before an RL agent executes its action, pass it through an LLM safety layer (DeepSeek/Ollama) that vetoes dangerous moves (e.g., collisions). Teaches value alignment and LLM-based safety filters.

### Language-to-Policy via Ollama Finetuning
Use few-shot learning or lightweight fine-tuning on Ollama to directly map textual intentions to policies: “Try to get fruit fast, but avoid walls.” Students compare LLM-generated policies vs. traditional RL.

可以做的对比：
- LLM 生成的策略：用 Ollama 直接根据自然语言生成动作或策略，不需要传统 RL 的大量训练。
- 传统强化学习策略：用 RL 算法（比如 DQN、PPO）通过大量试错训练学到的策略。


###  Graph Neural Networks for Snake 


- Snake的身体是由顺序连接的节点（身体段）组成，形成一条链状结构，天然就是图（节点+边）。
- 空间关系（蛇头与水果的距离和方向、蛇头与墙壁/自身身体的接近程度）可以通过节点间的邻接关系和边特征（如相对位置向量）精确建模。
- 动态性（蛇移动、水果被吃、身体增长）可以高效地通过动态图更新（添加/删除节点/边，更新节点特征）来实现。


- GNN的核心是消息传递机制。每个节点（蛇头、身体段、水果）可以聚合来自邻居（相邻的格子、连接的身体段）的信息。这使得模型能直接学习到：
- 蛇头下一步移动的安全性（是否会撞墙或撞自己？邻居节点是墙/身体吗？）。
- 蛇头到水果的最优路径方向（哪个方向的邻居更靠近水果？路径是否安全？）。
- 全局状态的理解（通过多轮消息传递，信息可以从蛇尾传播到蛇头，或从水果传播到蛇头）。

**结论:** 使用图神经网络 (GNN) 来建模和控制贪吃蛇游戏是一个高度可行且技术优势明显的方案。 它完美契合了游戏的图结构本质，能够高效地捕捉蛇的身体连接、与水果/墙壁的空间关系，并通过强化学习训练出强大的决策智能体。

### Emergent Behaviors via Multi-LLM Ensembles
Run multiple LLMs (Ollama, DeepSeek, Mixtral) in parallel, with varying prompts, and vote or combine their outputs. Analyze ensemble behavior, consistency, and divergence—teaches AI alignment and robustness.

### Self-Improving Snake via Prompt Tuning
Have a feedback loop where the LLM rewrites its own prompts based on game performance logs:

Before: “Avoid walls.”

After: “Avoid wall collisions near corners.”

This is a self-evolving LLM agent, which demonstrates prompt evolution and adaptation.

### Chain-of-Thought Navigation
Use Ollama/DeepSeek to generate step-by-step reasoning behind move decisions. Compare CoT reasoning vs. direct policy predictions and evaluate which leads to more consistent success.

### LLM-as-teacher, Distillation of LLM policy	

不一定非要人手写 heuristic，也可以让大语言模型（LLM）来扮演“老师”，生成这些示例策略数据。

大语言模型（如 GPT-4、Mistral、Claude）担任“教师”角色，生成状态下应该采取的动作（策略样本）。

把 LLM 生成的策略（动作决策）作为“软标签”，让一个更小的模型（如 MLP、CNN、小型 Transformer）去模仿，称为“策略蒸馏”。

### Heuristics 生成的数据来 引导/Distillation/fine tune LLM models

绝对可以！而且这其实是**近年来非常有效的训练范式之一**，可以总结为：

> ✅ **用 heuristic 策略生成高质量数据，引导、蒸馏（Distillation）、或微调（Fine-tune）LLM，使其具备领域智能或策略能力。**

---

### 🎯 为什么用 heuristic 来指导 LLM 是合理的？

| 原因                      | 描述                                   |
| ----------------------- | -------------------------- |
| ✅ Heuristic 是结构化、稳定、可控的 | 它能提供大量正确、高质量的动作样本，避免 LLM“胡说”。        |
| ✅ LLM 对状态-动作的理解可以迁移泛化   | 微调后，LLM 可能在新状态上生成更优的动作，甚至推理式动作。      |
| ✅ 比人类标注更省成本             | 自动生成的策略数据可以成千上万条，比人工标注便宜得多。          |
| ✅ 可持续增强                 | 可以不断迭代地用更好的 heuristic 生成更优的数据集来提升模型。 |

---

## 🧠 三种常见做法对比：

| 方法                         | 描述                                                  |
| -------------------------- | ----------------- |
| **Fine-tuning**            | 用 heuristic 生成的 `(状态, 文本动作)` 数据微调 LLM，让它学会“如何表述动作”。 |
| **Distillation**           | 用 heuristic 决策替代人类专家，用来训练 LLM 模仿决策风格。               |
| **Reinforcement Learning** | 用 heuristic reward 指导 LLM 在 RL 环境中强化训练（比如 RLAIF）。   |

---

### ✅ 示例：Snake Game 中 distill heuristic into LLM

假设你有如下 heuristic 策略：

> “蛇应该优先靠近食物，但不能撞墙。”

你可以自动生成状态和动作：

```json
{
  "state": {
    "snake_head": [5, 5],
    "food": [7, 5],
    "walls": [[5,6]]
  },
  "action": "move down"
}
```

然后构造成 Prompt → Response 的微调样本：

```
Prompt: 当前蛇头在(5,5)，食物在(7,5)，墙在(5,6)。请问下一步动作？
Response: 向下移动（move down），这样可以接近食物并避免碰墙。
```

成千上万个这样的数据可以：

* 微调一个 LLM 变得更“聪明”；
* 或蒸馏为一个轻量版 ChatModel for Snake。

---

### ✅ 更进一步的组合：

| 技术名词                              | 说明                                                |
| -------------------------- | --------------- |
| **Self-Instruct with Heuristics** | 用启发式生成大量 instruction-style 数据，像训练 Alpaca 一样训练小模型。 |
| **Reward Tuning via Heuristic**   | 用启发式 reward（如离食物远就惩罚）来训练 LLM 用于强化学习（见 Eureka）。    |
| **Curriculum Distillation**       | 从简单状态逐步生成难度更高状态，用 heuristic 指导模型训练路径。             |

