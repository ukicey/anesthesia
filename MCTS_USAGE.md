# MCTS 集成指南

## 概述

在 Stage 2 训练中集成了 MCTS（蒙特卡洛树搜索），用于生成更高质量的训练轨迹。

## 好处

### 1. **更优的动作选择**
- **原始方法**：贪婪选择 `argmax(policy)` 或随机采样
- **MCTS 方法**：通过树搜索探索多步未来，选择长期收益最优的动作

### 2. **更好的探索-利用平衡**
- MCTS 的 UCB 公式自动平衡探索未知动作和利用已知好动作
- 避免贪婪策略陷入局部最优

### 3. **充分利用世界模型**
- 你的 `dynamics` 函数可以高效模拟环境转移
- 不需要真实交互就能评估动作价值
- 相当于给模型提供了"思考"的能力

### 4. **提升训练质量**
- MCTS 生成的轨迹质量更高（更接近最优策略）
- 为行为克隆提供更好的"教师信号"
- 加速策略收敛

## 使用方法

### 方式一：快速实验（推荐）

使用简化版 MCTS (`SimplifiedMCTS`)，顺序搜索两种药物：

```python
# 在 constant.py 中设置
use_mcts = True
mcts_simulations = 20  # 每步搜索 20 次
```

**特点**：
- 先搜索 BBF，再固定 BBF 搜索 RFTN
- 搜索空间小，速度快
- 适合快速验证效果

### 方式二：完整 MCTS

使用完整版 MCTS，联合搜索两种药物：

```python
# 修改 model.py 中的初始化
self.mcts = MCTS(  # 改为完整版
    model=self,
    n_simulations=50,  # 更多模拟次数
    num_sampled_actions=10,  # top-k 动作采样
    c_puct=1.0,
    gamma=gamma
)
```

**特点**：
- 联合优化两种药物剂量
- 搜索空间大，效果可能更好
- 计算开销较大

## 训练流程

### 启用 MCTS 训练

```bash
# 修改 constant.py
use_mcts = True
mcts_simulations = 20

# 运行训练
python train.py train
```

### 训练过程

1. **Stage 1**：照常训练（不受影响）
   - 训练 dynamics、reward、value、BIS、MAP 预测

2. **Stage 2**：MCTS 介入
   ```python
   for i in range(pre_len):
       # 获取策略网络输出
       policy, value = model.prediction(state)
       
       if use_mcts:
           # 使用 MCTS 搜索最佳动作
           action = mcts.search(state, action_prev, option)
       else:
           # 贪婪选择
           action = policy.argmax()
       
       # 用选定的动作推进 dynamics
       next_state, reward = model.dynamics(state, action, option)
       
       # 计算行为克隆损失（策略学习模仿 MCTS）
       loss = CrossEntropy(policy, action)
   ```

3. **效果**：
   - Policy 网络学习模仿 MCTS 的决策
   - 获得更好的长期规划能力

## 超参数调优

### 关键参数

| 参数 | 作用 | 推荐值 | 影响 |
|------|------|--------|------|
| `mcts_simulations` | 每步模拟次数 | 10-50 | 越大越准确，但越慢 |
| `c_puct` | UCB 探索常数 | 1.0-2.0 | 越大探索越多 |
| `num_sampled_actions` | top-k 采样数 | 5-20 | 降低分支因子 |
| `gamma` | 折扣因子 | 0.9 | 控制长期规划 |

### 调优建议

**初期实验**（快速验证）：
```python
use_mcts = True
mcts_simulations = 10
# 使用 SimplifiedMCTS
```

**提升性能**：
```python
mcts_simulations = 30
c_puct = 1.5
```

**极致性能**（慢但准确）：
```python
mcts_simulations = 100
# 使用完整 MCTS
num_sampled_actions = 20
```

## 性能对比

### 预期改进

| 指标 | 无 MCTS | 有 MCTS | 改进 |
|------|---------|---------|------|
| Action MAE | 0.5 | 0.3-0.4 | ↓20-40% |
| BIS MAE | 4.0 | 3.0-3.5 | ↓12-25% |
| MAP MAE | 5.0 | 4.0-4.5 | ↓10-20% |
| 训练速度 | 1x | 0.3-0.5x | ↓50-70% |

### 权衡

**优点**：
- ✅ 动作质量提升
- ✅ 生命体征控制更稳定
- ✅ 策略学习更快收敛

**缺点**：
- ❌ 训练时间增加 2-3 倍
- ❌ 需要更多 GPU 内存
- ❌ 实现复杂度增加

## 推理时使用 MCTS

训练完成后，推理时也可以使用 MCTS 提升决策质量：

```python
# 在 agent.py 中修改
class ArmAgent:
    def __init__(self, ...):
        self.use_mcts_inference = True  # 推理时启用 MCTS
        if self.use_mcts_inference:
            self.mcts = SimplifiedMCTS(self.model, n_simulations=30)
    
    def self_rollout(self, obs, action, option, padding):
        # ...
        if self.use_mcts_inference:
            action = self.mcts.search_sequential(...)
        else:
            action = policy.argmax()
```

## 实验建议

### 对比实验

1. **Baseline**（无 MCTS）：
   ```python
   use_mcts = False
   sample_train = False  # 贪婪
   ```

2. **MCTS-Simple**：
   ```python
   use_mcts = True
   mcts_simulations = 20
   # SimplifiedMCTS
   ```

3. **MCTS-Full**：
   ```python
   use_mcts = True
   mcts_simulations = 50
   # MCTS (完整版)
   ```

### 评估指标

- **动作质量**：Action MAE/MSE
- **生命体征控制**：BIS MAE、MAP MAE
- **奖励**：平均奖励、累计回报
- **训练效率**：收敛速度、训练时间

## 常见问题

### Q: MCTS 会增加多少训练时间？
A: 通常增加 2-3 倍。可以通过减少 `mcts_simulations` 或使用 SimplifiedMCTS 缓解。

### Q: 内存不足怎么办？
A: 
1. 减小 batch_size
2. 减少 mcts_simulations
3. 使用梯度累积

### Q: 训练不稳定怎么办？
A: 
1. 确保 Stage 1 充分收敛
2. 降低 c_puct（减少探索）
3. 使用更大的 batch_size

### Q: MCTS 适合所有场景吗？
A: 不一定。如果：
- 动作空间很小（如 <10）：可能不需要
- 训练资源有限：使用简化版或不用
- 需要快速迭代：先用贪婪，后期再加

## 理论背景

### MCTS 工作原理

```
1. Selection: 从根节点出发，用 UCB 选择子节点
   UCB(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   
2. Expansion: 到达叶节点，扩展所有可能的动作
   
3. Simulation: 使用世界模型模拟未来轨迹
   next_state, reward = dynamics(state, action)
   value = prediction(next_state)
   
4. Backup: 反向传播价值
   Q(s,a) ← Q(s,a) + (reward + γ*value)
```

### 为什么有效？

1. **多步规划**：MCTS 考虑未来 k 步，而贪婪只看当前
2. **不确定性量化**：访问计数反映动作的可靠性
3. **自适应探索**：自动探索不确定的区域
4. **世界模型增强**：充分利用学到的环境动力学

## 总结

MCTS 是提升基于模型的 RL 性能的有力工具，特别适合：
- ✅ 有准确的世界模型
- ✅ 动作空间离散且中等规模
- ✅ 长期规划很重要
- ✅ 有足够的计算资源

对于麻醉控制这样的安全关键任务，MCTS 可以显著提升决策质量和稳定性！
