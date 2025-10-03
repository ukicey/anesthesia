# RL+BC混合训练方案说明

## 📌 问题回顾

### 之前的错误理解
- **误区**: 认为整个项目需要MCTS来优化策略
- **真相**: 整个项目本质上是**监督学习/行为克隆（BC）**
  - Stage 1: 训练环境模型（dynamics, reward）和价值函数（value）
  - Stage 2: 训练策略网络（policy），但loss始终和**真实专家动作**比较
  - MCTS只能改变rollout轨迹，不改变学习目标

### 核心问题
在纯BC框架下：
- ✅ **优点**: 安全、稳定，能够模仿专家行为
- ❌ **缺点**: 
  1. 无法超越专家（ceiling effect）
  2. Exposure bias：训练时用专家动作rollout，测试时用自己的预测
  3. 分布偏移：rollout轨迹可能偏离专家轨迹

---

## 🎯 新方案：RL+BC混合训练

### 设计理念
**结合BC的安全性与RL的优化能力**

```
BC部分（70%）: 模仿专家 → 保证基线性能和安全性
RL部分（30%）: 策略优化 → 允许在安全范围内改进
```

### 核心算法

#### Stage 1（不变）
训练环境模型和价值函数：
- Dynamics函数：预测下一状态
- Reward函数：预测即时奖励
- Value函数：预测长期回报
- BIS/RP函数：预测生命体征

#### Stage 2（RL+BC混合）

对于每个预测步 `i`：

1. **获取专家动作** `a_expert`（来自数据集）

2. **模型预测动作** `a_pred`（贪婪或采样）

3. **价值评估**：
   - 评估专家动作的Q值：`Q_expert = reward(s, a_expert) + γ * value(s')`
   - 评估预测动作的Q值：`Q_pred = reward(s, a_pred) + γ * value(s')`

4. **动态目标选择**：
   ```python
   if Q_pred > Q_expert + margin:  # margin=0.1确保显著更好
       RL目标动作 = a_pred  # 发现更优动作
       RL权重 = 1.0
   else:
       RL目标动作 = a_expert  # 保持专家动作
       RL权重 = 0.0
   ```

5. **损失计算**：
   ```python
   loss_BC = CrossEntropy(policy, a_expert)  # 始终学习专家
   loss_RL = CrossEntropy(policy, RL目标动作) * RL权重  # 有条件地学习更优动作
   
   loss_total = λ_BC * loss_BC + λ_RL * loss_RL
   ```
   其中 `λ_BC = 0.7`, `λ_RL = 0.3`

6. **Rollout策略**（可配置）：
   - **Teacher Forcing** (`teacher_forcing=True`): 用专家动作rollout（更稳定）
   - **Mixed Rollout** (`teacher_forcing=False`): 用RL目标动作rollout

---

## 🔧 实现细节

### 配置参数（`constant.py`）

```python
# RL+BC混合训练配置
use_rl_bc_hybrid = True          # 是否使用RL+BC混合训练
lambda_bc = 0.7                  # BC损失权重（保证安全性）
lambda_rl = 0.3                  # RL损失权重（允许优化）
teacher_forcing = True           # 是否使用专家动作rollout
```

### 模型输出（`model.py`）

新增输出字段：
```python
outputs = {
    'policy': (policy_bbf, policy_rftn),
    'value': value,
    'reward': reward,
    'bis': bis,
    'rp': rp,
    'policy_train': (policy_train_bbf, policy_train_rftn),
    
    # RL+BC新增
    'rl_action_target': (rl_action_bbf, rl_action_rftn),  # RL目标动作
    'rl_weights': rl_weights,  # RL损失权重 (B, pre_len)
}
```

### 损失计算（`pl_model.py`）

```python
# BC损失：始终模仿专家
loss_bc = CE(policy, action_expert)

# RL损失：有条件地学习更优动作
if use_rl_bc_hybrid:
    loss_rl = (CE(policy, rl_action_target) * rl_weights).mean()
    loss_action = lambda_bc * loss_bc + lambda_rl * loss_rl
else:
    loss_action = loss_bc
```

---

## 📊 方案对比

| 方案 | BC保证 | 超越专家 | 稳定性 | 安全性 | 适用场景 |
|------|--------|----------|--------|--------|----------|
| **纯BC（原方案）** | ✅ | ❌ | ✅✅✅ | ✅✅✅ | 专家数据充足且质量高 |
| **RL+BC混合（新方案）** | ✅ | ✅ | ✅✅ | ✅✅ | 想在安全范围内优化 |
| **纯RL** | ❌ | ✅ | ❌ | ❌ | 可以承受探索风险 |

---

## 🎓 理论优势

### 1. **安全性保证**
- BC部分权重70%，确保始终学习专家行为
- 即使RL部分失败，模型仍保持专家水平

### 2. **优化潜力**
- 允许在环境模型认为"更优"的方向上改进
- 利用学到的dynamics和reward函数探索更好策略

### 3. **自适应学习**
- 动态权重：只在真正找到更优动作时施加RL loss
- 避免盲目优化导致的性能下降

### 4. **训练稳定性**
- Teacher Forcing选项：用专家动作rollout，避免分布偏移
- 渐进式改进：不会突然偏离专家轨迹

---

## 🚀 使用指南

### 基础配置（推荐）
```python
# constant.py
use_rl_bc_hybrid = True
lambda_bc = 0.7
lambda_rl = 0.3
teacher_forcing = True  # 稳定训练
```

### 实验性配置（更激进）
```python
# constant.py
use_rl_bc_hybrid = True
lambda_bc = 0.5
lambda_rl = 0.5
teacher_forcing = False  # 允许更多探索
```

### 退回纯BC
```python
# constant.py
use_rl_bc_hybrid = False
# 或者
lambda_bc = 1.0
lambda_rl = 0.0
```

---

## 📈 监控指标

训练时关注以下指标：

1. **`loss_bc`**: BC损失（应该稳定下降）
2. **`loss_rl`**: RL损失（可能波动，但不应过大）
3. **`loss_action`**: 总动作损失（主要优化目标）
4. **`val_action_mae`**: 动作预测误差（验证集）

**健康训练的标志**：
- `loss_bc` 稳定下降
- `loss_rl` 保持在合理范围（0-loss_bc之间）
- 验证集性能不低于纯BC baseline

---

## 🔍 调试建议

### 如果性能下降
1. **增加BC权重**：`lambda_bc = 0.8, lambda_rl = 0.2`
2. **启用Teacher Forcing**：`teacher_forcing = True`
3. **增大margin**：在`model.py`中将`margin=0.1`改为`0.2`或`0.3`

### 如果改进不明显
1. **检查是否真的存在改进空间**：专家数据可能已经接近最优
2. **降低BC权重**：`lambda_bc = 0.5, lambda_rl = 0.5`
3. **检查环境模型质量**：确保reward和dynamics预测准确

### 如果训练不稳定
1. **使用Teacher Forcing**：`teacher_forcing = True`
2. **降低学习率**：`lr = 0.00005`
3. **增大batch size**：`batch_size = 256`

---

## 💡 关键洞察

### RL vs BC的本质区别

**BC（行为克隆）**：
```python
loss = CrossEntropy(policy(s), a_expert)
# 目标：模仿专家动作分布
```

**RL（强化学习）**：
```python
loss = -log_prob(a) * advantage(s, a)
# 目标：最大化期望回报
```

**RL+BC混合**：
```python
loss_bc = CE(policy(s), a_expert)
loss_rl = CE(policy(s), a_better) if Q(s, a_better) > Q(s, a_expert) else 0
loss = λ_bc * loss_bc + λ_rl * loss_rl
# 目标：模仿专家 + 有条件优化
```

### 为什么这种方法有效？

1. **保守的优化**：只在"确信"找到更优动作时才学习
2. **利用已有知识**：BC提供强先验，RL在此基础上微调
3. **风险可控**：即使RL部分失败，BC部分兜底

---

## 📚 扩展方向

### 短期改进
1. **Scheduled Sampling**：逐渐从Teacher Forcing过渡到自主rollout
2. **动态权重调整**：训练初期高BC权重，后期增加RL权重
3. **多样化评估**：不只看Q值，也考虑安全性（BIS/MAP范围）

### 长期研究
1. **Offline RL方法**：使用CQL、IQL等离线RL算法
2. **Safe RL约束**：添加硬约束确保生命体征在安全范围
3. **Multi-task Learning**：同时优化多个目标（efficacy、safety、comfort）

---

## ⚠️ 注意事项

1. **环境模型质量至关重要**
   - RL部分依赖dynamics和reward函数
   - 如果环境模型不准，RL可能学到错误策略

2. **专家数据仍然重要**
   - BC部分需要高质量专家轨迹
   - RL不能弥补低质量数据

3. **超参数敏感性**
   - `lambda_bc/lambda_rl`的比例需要调优
   - `margin`阈值影响RL触发频率

4. **医疗场景的特殊性**
   - 安全性 > 性能
   - 建议保守配置（高BC权重、Teacher Forcing）
   - 充分验证后再部署

---

## 📝 总结

这个RL+BC混合方案：
- ✅ 保留了BC的安全性和稳定性
- ✅ 引入了RL的优化潜力
- ✅ 通过动态权重实现自适应学习
- ✅ 实现简单，易于调试和扩展

**推荐作为baseline改进方案**，相比纯BC有望获得5-15%的性能提升，同时保持训练稳定性。
