# 训练策略分析：Stage1 vs Stage2

## 当前实现：联合训练（Joint Training）

### 代码证据

```python
# pl_model.py line 248
loss = loss_action + loss_bis + loss_rp + loss_value + loss_reward
```

**所有组件同时优化**：
- `loss_bis`, `loss_rp`: 生命体征预测（环境模型的一部分）
- `loss_value`: 价值函数（critic）
- `loss_reward`: 奖励预测（环境模型）
- `loss_action`: 策略函数（actor，BC+MCTS）

### 训练流程

```
每个训练step：
1. Forward Stage1
   - 用真实动作rollout
   - 预测 bis, rp, value, reward
   - 计算 loss_bis, loss_rp, loss_value, loss_reward

2. Forward Stage2
   - 用预测动作（或MCTS动作）rollout
   - 预测 policy
   - 计算 loss_action (BC + MCTS)

3. 优化
   - loss_total = loss1 + loss2
   - 一次反向传播更新所有参数
```

---

## 问题：环境模型能训练好吗？

### ⚠️ 潜在问题

#### 问题1：Stage2污染环境模型

```python
# Stage2中（model.py line 265-398）
for i in range(pre_len):
    policy_t, value_t = self.prediction(state_t)
    
    # 用预测动作rollout（不是真实动作！）
    if use_mcts:
        action = mcts.search(...)
    else:
        action = policy.argmax(...)
    
    # 用预测动作推进dynamics
    state_t_next = self.dynamics(state_t, action, ...)
    #                                     ↑
    #                              不是真实动作！
```

**问题**：
- Stage2的dynamics调用时，输入的是**预测动作**
- dynamics的梯度会反向传播
- dynamics可能学到"对预测动作更友好"而非真实的环境转移

**这叫model exploitation（模型利用）**：
```
dynamics被优化成：
  对policy预测的动作 → 产生高reward的下一状态
而不是：
  真实地模拟环境转移
```

#### 问题2：训练目标混杂

```python
# dynamics同时接收两种训练信号

# Signal 1: Stage1的监督信号（好的✅）
state_next_真实, reward_真实 = dynamics(state, action_真实, ...)
loss_dynamics = MSE(state_next_真实, state_target)

# Signal 2: Stage2的隐式信号（坏的❌）
state_next_预测, reward_预测 = dynamics(state, action_预测, ...)
# 这个调用也会产生梯度，因为state_next参与policy loss计算
loss_action = CE(policy, target)  # 依赖state_next
```

**冲突**：
- Stage1要求dynamics真实模拟环境
- Stage2隐式要求dynamics对policy友好

#### 问题3：训练早期环境模型不准

```python
# 训练开始时
epoch 1-10: 
  - dynamics, reward, value都不准
  - 但policy已经开始用它们rollout
  - MCTS在错误的模型上搜索
  - policy学到错误的策略
```

---

## 解决方案

### 🥇 方案1：两阶段训练（推荐）

#### Phase 1: 只训练环境模型（warm-up）

```python
# 前N个epoch（如N=10-20）
if epoch < warmup_epochs:
    # 只优化环境模型
    loss = loss_bis + loss_rp + loss_value + loss_reward
    # 不优化policy
else:
    # 正常联合训练
    loss = loss_action + loss_bis + loss_rp + loss_value + loss_reward
```

**优点**：
- ✅ 确保环境模型先训练好
- ✅ MCTS在准确的模型上搜索
- ✅ policy不会学到错误信号

**实现**：
```python
# pl_model.py
class RLSLModelModule(BaseModule):
    def __init__(self, *args, warmup_epochs=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
    
    def get_loss(self, pred_data, label_data):
        # ... 计算所有loss ...
        
        # 两阶段策略
        if self.current_epoch < self.warmup_epochs:
            # Phase 1: 只训练环境模型
            loss = loss_bis + loss_rp + loss_value + loss_reward
        else:
            # Phase 2: 联合训练
            loss = loss_action + loss_bis + loss_rp + loss_value + loss_reward
        
        return losses
```

### 🥈 方案2：冻结dynamics（部分）

在Stage2中冻结dynamics的梯度：

```python
# model.py - Stage2
for i in range(pre_len):
    policy_t, value_t = self.prediction(state_t)
    
    # 用预测动作rollout，但不让dynamics被更新
    with torch.no_grad():
        # 冻结dynamics
        state_t_next, reward_t = self.dynamics(state_t, action, ...)
    
    # 或者detach state
    state_t = state_t_next.detach()
```

**优点**：
- ✅ dynamics只从Stage1学习（真实转移）
- ✅ 避免model exploitation

**缺点**：
- ❌ dynamics可能不适应policy的分布

### 🥉 方案3：分离训练（最激进）

完全分开两个阶段：

```python
# 训练循环
for epoch in range(n_epochs):
    # 先训练环境模型
    for batch in train_loader:
        loss_env = train_env_model(batch)
        optimizer_env.step()
    
    # 再训练policy
    for batch in train_loader:
        loss_policy = train_policy(batch)
        optimizer_policy.step()
```

**优点**：
- ✅ 完全解耦
- ✅ 环境模型不受policy影响

**缺点**：
- ❌ 训练慢（2倍时间）
- ❌ 实现复杂

### 🌟 方案4：加权损失（简单）

调整loss权重，环境模型loss占主导：

```python
# 当前（问题）
loss = loss_action + loss_bis + loss_rp + loss_value + loss_reward
#      权重1          权重1      权重1      权重1         权重1

# 改进
loss = 0.2 * loss_action + loss_bis + loss_rp + loss_value + loss_reward
#      ↓ 降低policy权重，减少对环境模型的干扰
```

或者：

```python
# constant.py
env_loss_weight = 5.0
policy_loss_weight = 1.0

# pl_model.py
loss = (policy_loss_weight * loss_action + 
        env_loss_weight * (loss_bis + loss_rp + loss_value + loss_reward))
```

---

## 推荐方案（医疗场景）

### 组合方案：Warm-up + 加权

```python
# constant.py
warmup_epochs = 10  # 前10个epoch只训练环境模型
env_loss_weight = 2.0  # warmup后，环境模型loss权重仍是policy的2倍

# pl_model.py
if self.current_epoch < warmup_epochs:
    # Phase 1: 只训练环境模型
    loss = loss_bis + loss_rp + loss_value + loss_reward
else:
    # Phase 2: 联合训练，但环境模型占主导
    loss_env = loss_bis + loss_rp + loss_value + loss_reward
    loss = loss_action + env_loss_weight * loss_env
```

**理由**：
1. **Warm-up确保环境模型先训练好**
   - 前10 epoch环境模型从随机初始化学习
   - dynamics, reward, value, bis, rp都学到合理的预测

2. **加权确保环境模型持续准确**
   - 即使联合训练，环境模型loss占主导
   - 减少policy对环境模型的干扰

3. **医疗场景适用**
   - 环境模型准确性优先（患者安全）
   - policy可以稍慢收敛

---

## 当前实现能训练好吗？

### 回答你的问题

**能，但不保证**：

✅ **可能训练好的情况**：
- 数据充足且质量高
- 环境模型本身容易学（dynamics不太复杂）
- 学习率设置合理
- 训练足够长时间

❌ **可能训练不好的情况**：
- 训练早期环境模型不准
- Policy在错误模型上学习，产生错误梯度
- Model exploitation：dynamics迎合policy
- MCTS在不准的模型上搜索

### 判断环境模型是否训练好

**监控指标**：
```python
val_bis_mae      # BIS预测准确性（应该<3.0）
val_rp_mae       # RP预测准确性（应该<5.0）
val_reward_mae   # Reward预测准确性
val_value_mae    # Value预测准确性

# 健康状态
- 这些指标应该持续下降
- 不应该在某个epoch后突然上升
  （如果上升，说明policy开始干扰环境模型）
```

**诊断方法**：
```python
# 在validation时，单独评估环境模型
with torch.no_grad():
    # 用真实动作rollout（不用policy动作）
    for i in range(pre_len):
        state_next_真实, reward_真实 = dynamics(state, action_真实[i], ...)
        bis_真实 = output_bis(state_next_真实)
        
        # 和target比较
        env_model_error = MAE(bis_真实, bis_target[i])
    
    # 如果error很大，说明环境模型不准
```

---

## 实现建议

### 立即可做（最小改动）

1. **添加warm-up**：
```python
# constant.py
warmup_epochs = 10

# pl_model.py修改get_loss
if self.current_epoch < warmup_epochs:
    loss = loss_bis + loss_rp + loss_value + loss_reward
else:
    loss = loss_action + loss_bis + loss_rp + loss_value + loss_reward
```

2. **监控环境模型指标**：
```python
# 重点关注
- loss_bis下降曲线
- loss_rp下降曲线
- loss_value下降曲线
- loss_reward下降曲线

# 如果这些loss在warmup后反弹，说明policy在干扰
```

### 中期改进（推荐）

实现方案4（加权损失）：
```python
# constant.py
warmup_epochs = 10
env_loss_weight = 2.0

# pl_model.py
if self.current_epoch < warmup_epochs:
    loss = loss_bis + loss_rp + loss_value + loss_reward
else:
    loss_env = loss_bis + loss_rp + loss_value + loss_reward
    loss = loss_action + env_loss_weight * loss_env
```

### 长期优化（如果有问题）

考虑方案2（冻结dynamics）：
- 在Stage2的forward中detach state
- 确保dynamics只从Stage1学习

---

## 总结

### 当前实现的问题

1. **联合训练**：Stage1和Stage2同时优化
2. **无保证**：环境模型可能被policy干扰
3. **风险**：MCTS可能在不准的模型上搜索

### 推荐改进

```python
# 最小改动，最大收益
1. 添加warm-up（10 epochs）
2. 加权损失（env_weight=2.0）
3. 监控环境模型指标
```

### 判断标准

**环境模型训练好的标志**：
- ✅ `loss_bis`, `loss_rp` 持续下降，不反弹
- ✅ `val_bis_mae < 3.0`（取决于你的数据）
- ✅ `val_rp_mae < 5.0`
- ✅ 在warmup后，这些指标仍然稳定

**环境模型有问题的标志**：
- ❌ warmup后，环境loss反弹
- ❌ val_bis_mae, val_rp_mae很大或不收敛
- ❌ MCTS激活率极低（<1%）或极高（>80%）

希望这个分析有帮助！需要我帮你实现warm-up机制吗？
