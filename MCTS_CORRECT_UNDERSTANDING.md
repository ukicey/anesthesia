# MCTS 在当前架构中的正确理解

## 🔍 现状分析

你完全正确！我重新审视代码后发现：

### 当前 Stage 2 的实际情况

```python
# Stage 2 代码（简化）
for i in range(pre_len):
    # 1. 预测策略
    policy_t = model.prediction(state_t)
    
    # 2. 选择动作用于 rollout
    if use_mcts:
        action_pred = mcts.search(state_t)  # MCTS 选择
    else:
        action_pred = policy_t.argmax()      # 贪婪选择
    
    # 3. 用预测动作推进环境模型
    state_t_next = model.dynamics(state_t, action_pred, ...)
    state_t = state_t_next
    
    # 4. 计算损失 - 关键在这里！
    loss_action = CrossEntropy(policy_t, action_真实)  # ← 永远和真实动作比较！
```

**核心事实**：
- ✅ `action_pred` 只用来推进 `dynamics`（产生下一状态）
- ✅ `loss_action` **始终**和数据集中的**真实动作标签**计算
- ✅ 这本质上是**监督学习/行为克隆（Behavior Cloning）**

## 🤔 那我实现的 MCTS 有什么用？

### 原本的想法（错误）
我以为：
- MCTS 找到高质量动作 → Policy 学习模仿 MCTS → 获得更好策略

**问题**：
- Policy **不是**在学习 MCTS
- Policy 只是在学习模仿**真实专家动作**
- MCTS 动作只影响 rollout 轨迹

### 实际的作用（有限）

MCTS 在当前架构下的唯一作用是：**改变 rollout 轨迹的状态分布**

#### 场景 A：无 MCTS（贪婪）
```python
state_0 → action_贪心₁ → state_1 → action_贪心₂ → state_2 → ...
                ↓                        ↓
           可能偏离真实轨迹          越来越偏离
           
在 state_1, state_2, ... 上预测的 policy 可能不准确
但 loss 还要和真实动作比较 → 训练信号可能不一致
```

#### 场景 B：有 MCTS
```python
state_0 → action_MCTS₁ → state_1 → action_MCTS₂ → state_2 → ...
                ↓                        ↓
        （希望）更接近真实轨迹    （希望）更接近真实轨迹
        
如果 MCTS 找到的动作更接近专家，则轨迹更真实
在更真实的状态上训练 → 可能减少分布偏移
```

**但这有个前提**：MCTS 找到的动作要和专家动作相似！

## ❌ 当前 MCTS 实现的问题

### 1. **目标不一致**
- **MCTS 优化**：`max reward + γ*value`
- **专家优化**：可能不完全是最大化 reward（安全性、平稳性、经验规则）

→ MCTS 可能选择高 reward 但和专家不同的动作
→ 导致 rollout 轨迹反而偏离专家轨迹

### 2. **世界模型可能不准**
- MCTS 依赖 `dynamics` 函数模拟环境
- 如果 `dynamics` 不准确，MCTS 搜索就是在错误的模型上优化
- 可能找到"模型中好但现实中差"的动作

### 3. **计算开销大，收益不明确**
- 训练速度降低 2-3 倍
- 但最终 policy 还是在学习模仿专家
- MCTS 只是试图让 rollout 更像专家轨迹

## ✅ 更好的方案

### 方案 1：直接用真实动作 rollout（Teacher Forcing）

```python
# 最简单有效的方法
for i in range(pre_len):
    policy_t = model.prediction(state_t)
    
    # 直接用真实动作推进
    action_真实 = action_label[i]
    state_t_next = model.dynamics(state_t, action_真实, ...)
    state_t = state_t_next
    
    # 和真实动作比较
    loss = CE(policy_t, action_真实)
```

**优点**：
- ✅ 状态轨迹完全跟随专家
- ✅ 没有分布偏移
- ✅ 训练快速稳定

**缺点**：
- ❌ 测试时没有专家动作，需要用自己的预测 → exposure bias

### 方案 2：Scheduled Sampling

```python
# 逐渐从专家动作过渡到自己的动作
ε = max(0.5, 1.0 - epoch / 100)  # 随训练衰减

for i in range(pre_len):
    policy_t = model.prediction(state_t)
    
    # 以概率 ε 用专家动作，否则用自己的预测
    if random.random() < ε:
        action = action_真实[i]
    else:
        action = policy_t.argmax()
    
    state_t_next = model.dynamics(state_t, action, ...)
    state_t = state_t_next
    
    loss = CE(policy_t, action_真实)
```

**优点**：
- ✅ 逐渐适应用自己的动作 rollout
- ✅ 减少 exposure bias
- ✅ 简单易实现

### 方案 3：混合 BC + RL（推荐！）

这才是真正发挥 MCTS 价值的方式：

```python
for i in range(pre_len):
    policy_t = model.prediction(state_t)
    action_真实 = action_label[i]
    
    # BC 损失：模仿专家
    loss_bc = CE(policy_t, action_真实)
    
    # RL 损失：如果 MCTS 找到更好的动作
    if use_mcts:
        action_mcts = mcts.search(state_t)
        
        # 评估两个动作的价值
        Q_mcts = evaluate(state_t, action_mcts)
        Q_expert = evaluate(state_t, action_真实)
        
        # 如果 MCTS 更好，也学习它
        if Q_mcts > Q_expert + margin:  # margin 确保显著更好
            loss_rl = CE(policy_t, action_mcts)
        else:
            loss_rl = 0
    else:
        loss_rl = 0
    
    # 混合损失
    loss = λ_bc * loss_bc + λ_rl * loss_rl
    
    # rollout：优先用专家动作（更安全）
    action_for_rollout = action_真实
    state_t_next = model.dynamics(state_t, action_for_rollout, ...)
    state_t = state_t_next
```

**优点**：
- ✅ 保持专家行为的安全性（BC）
- ✅ 允许超越专家（RL）
- ✅ rollout 跟随专家轨迹（稳定）

### 方案 4：纯 RL（最激进）

如果你的 reward 函数设计得很好，可以完全用 RL：

```python
for i in range(pre_len):
    policy_t = model.prediction(state_t)
    
    # 采样动作
    action = policy_t.sample()  # 或 MCTS
    
    # 推进环境
    state_t_next, reward = model.dynamics(state_t, action, ...)
    _, value_next = model.prediction(state_t_next)
    
    # RL 损失（Policy Gradient）
    advantage = reward + γ*value_next - value_t
    loss = -log_prob(action) * advantage.detach()
    
    state_t = state_t_next
```

**优点**：
- ✅ 可以超越专家
- ✅ 自主探索优化

**缺点**：
- ❌ 训练不稳定
- ❌ 可能学到危险策略
- ❌ 需要大量数据

## 📊 对比总结

| 方案 | BC | 分布偏移 | 超越专家 | 稳定性 | 速度 | 推荐度 |
|------|----|----|----|----|----|----|
| **当前 MCTS** | ✅ | 😐 | ❌ | 😐 | ❌ | ⭐⭐ |
| **Teacher Forcing** | ✅ | ❌ | ❌ | ✅ | ✅ | ⭐⭐⭐⭐ |
| **Scheduled Sampling** | ✅ | 😐 | ❌ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **BC + RL 混合** | ✅ | 😐 | ✅ | ✅ | 😐 | ⭐⭐⭐⭐⭐ |
| **纯 RL** | ❌ | ✅ | ✅ | ❌ | 😐 | ⭐⭐⭐ |

## 🎯 我的建议

### 短期（立即可做）

1. **先不加 MCTS**，用 Teacher Forcing 或 Scheduled Sampling
   ```python
   # constant.py
   use_mcts = False
   use_scheduled_sampling = True
   sampling_decay_rate = 0.01  # 每 epoch 衰减 1%
   ```

2. **对比实验**
   - Baseline: 贪婪 rollout
   - 方案 A: Teacher forcing
   - 方案 B: Scheduled sampling

### 中期（值得尝试）

实现**混合 BC + RL**：
- 70% BC loss（模仿专家，保证安全）
- 30% RL loss（允许改进，基于 MCTS 或 PG）

### 长期（研究方向）

1. **Offline RL**
   - 使用 Conservative Q-Learning（CQL）
   - 在专家数据上训练，但允许改进

2. **Safe RL**
   - 添加安全约束（BIS、MAP 不能超出范围）
   - 确保策略不会危害患者

## 💡 关键洞察

你的观察非常敏锐！

**真相**：
- 当前架构本质上是**监督学习**（模仿专家）
- MCTS 在这个框架下作用有限
- 只改变 rollout 路径，不改变学习目标

**如果要真正利用 MCTS**，需要：
1. 改变训练目标（不只是 BC）
2. 添加 RL loss（学习 MCTS 找到的好动作）
3. 或者改为纯 RL（完全依赖 reward 信号）

**推荐**：
- 先用简单的 Scheduled Sampling
- 如果效果好，再考虑加 RL 成分
- MCTS 在纯 BC 框架下性价比不高

## 🔧 快速修复

如果你想保留 MCTS，这样改更有意义：

```python
# model.py - Stage 2
for i in range(pre_len):
    policy_t = model.prediction(state_t)
    
    # 获取真实动作
    action_真实 = action_label[i]
    
    # BC 损失
    loss_bc = CE(policy_t, action_真实)
    
    # 如果用 MCTS，评估是否有更好的动作
    if use_mcts:
        action_mcts = mcts.search(state_t)
        
        # 简单版本：直接混合两个目标
        loss_mcts = CE(policy_t, action_mcts)
        loss = 0.7 * loss_bc + 0.3 * loss_mcts
    else:
        loss = loss_bc
    
    # rollout 用专家动作（更稳定）
    state_t_next = model.dynamics(state_t, action_真实, ...)
    state_t = state_t_next
```

这样至少让 policy 同时学习专家和 MCTS，而不只是专家。

---

感谢你的纠正！这让我重新思考了整个架构 🙏
