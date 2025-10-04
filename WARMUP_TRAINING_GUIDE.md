# Warm-up训练策略使用指南

## 🎯 问题与解决方案

### 问题：环境模型可能训练不好

**原因**：
- Stage1（环境模型）和Stage2（policy）同时训练
- 训练早期环境模型不准，但policy已经在用它
- MCTS在错误的模型上搜索
- Policy可能干扰环境模型（model exploitation）

**解决方案**：Warm-up + 加权

```
Phase 1 (Warm-up):    只训练环境模型
Epoch 1-10            loss = loss_bis + loss_rp + loss_value + loss_reward
                      ↓
Phase 2 (Joint):      联合训练，环境模型权重更大
Epoch 11+             loss = loss_action + 2.0 * loss_env
```

---

## ⚙️ 配置参数

### `constant.py`中的新参数

```python
# 训练策略配置
warmup_epochs = 10      # 前10个epoch只训练环境模型
env_loss_weight = 2.0   # warmup后，环境模型loss权重是policy的2倍
```

### 推荐配置

#### 🥇 标准配置（推荐）
```python
warmup_epochs = 10
env_loss_weight = 2.0
```

#### 🥈 快速收敛（如果环境模型容易学）
```python
warmup_epochs = 5
env_loss_weight = 1.5
```

#### 🥉 保守配置（如果环境模型很难学）
```python
warmup_epochs = 20
env_loss_weight = 3.0
```

---

## 📊 训练过程

### Phase 1: Warm-up（Epoch 1-10）

**训练目标**：
```python
loss = loss_bis + loss_rp + loss_value + loss_reward
```

**优化组件**：
- ✅ Dynamics（状态转移）
- ✅ Reward函数
- ✅ Value函数（critic）
- ✅ BIS预测
- ✅ RP预测
- ❌ Policy（不训练）

**期望结果**：
- `loss_bis`下降到合理水平（<3.0）
- `loss_rp`下降到合理水平（<5.0）
- `loss_value`稳定下降
- `loss_reward`稳定下降

### Phase 2: Joint Training（Epoch 11+）

**训练目标**：
```python
loss_env = loss_bis + loss_rp + loss_value + loss_reward
loss = loss_action + 2.0 * loss_env
```

**优化组件**：
- ✅ 所有组件（环境模型 + policy）
- 环境模型loss权重是policy的2倍

**期望结果**：
- 环境模型继续准确（loss不反弹）
- Policy开始收敛
- `loss_mcts > 0`（MCTS开始激活）

---

## 📈 监控指标

### 关键指标

训练时重点关注：

```python
# Warmup阶段（Epoch 1-10）
in_warmup = 1.0          # 指示在warmup中
loss_bis ↓               # 应持续下降
loss_rp ↓                # 应持续下降
loss_value ↓             # 应持续下降
loss_reward ↓            # 应持续下降
loss_action = ?          # 会计算但不参与优化

# Joint训练阶段（Epoch 11+）
in_warmup = 0.0          # 离开warmup
loss_bis 持续↓           # 不应反弹！
loss_rp 持续↓            # 不应反弹！
loss_action ↓            # 开始下降
loss_mcts > 0            # MCTS激活
```

### 健康训练的标志

✅ **Phase 1结束时**：
```
Epoch 10:
  val_loss_bis: 2.5  (下降到合理水平)
  val_loss_rp: 4.2   (下降到合理水平)
  val_loss_value: 0.8
  val_loss_reward: 0.5
```

✅ **Phase 2开始后**：
```
Epoch 11-15:
  val_loss_bis: 2.5 → 2.3  (继续下降或稳定)
  val_loss_rp: 4.2 → 4.0   (继续下降或稳定)
  val_loss_action: 开始下降
  loss_mcts: > 0 (MCTS激活)
```

### ⚠️ 问题标志

❌ **环境模型有问题**：
```
Epoch 11-15:
  val_loss_bis: 2.5 → 3.5  ❌ 反弹！
  val_loss_rp: 4.2 → 5.8   ❌ 反弹！
```

**原因**：
- Policy干扰环境模型
- `env_loss_weight`太低

**解决**：
```python
env_loss_weight = 3.0  # 增大环境模型权重
```

---

## 🔍 诊断方法

### 检查环境模型质量

在validation时，可以单独评估环境模型：

```python
# 用真实动作rollout（不用policy动作）
with torch.no_grad():
    for i in range(pre_len):
        # 用真实专家动作
        state_next, reward = dynamics(state, action_真实[i], ...)
        bis_pred = output_bis(state_next)
        
        # 和target比较
        env_error = MAE(bis_pred, bis_target[i])
```

如果`env_error`很大（>5.0），说明环境模型不准。

### 对比warmup前后

```python
# 记录关键指标
Epoch 10 (warmup结束):
  val_bis_mae: 2.5
  val_rp_mae: 4.2

Epoch 20 (joint训练):
  val_bis_mae: 2.3  # 应该≤2.5
  val_rp_mae: 4.0   # 应该≤4.2

# 如果反而更差，说明有问题
```

---

## 🛠️ 调优建议

### 如果环境模型不够好

**症状**：
- `val_bis_mae > 3.0`持续不降
- `val_rp_mae > 6.0`持续不降
- MCTS激活率极低（<1%）

**解决方案**：

1. **延长warmup**：
```python
warmup_epochs = 20  # 或30
```

2. **增加环境模型权重**：
```python
env_loss_weight = 3.0  # 或5.0
```

3. **检查数据质量**：
   - 数据是否充足
   - 标签是否准确

### 如果训练太慢

**症状**：
- Policy收敛很慢
- 前10 epoch没有policy进展

**解决方案**：

1. **缩短warmup**：
```python
warmup_epochs = 5
```

2. **降低环境模型权重**：
```python
env_loss_weight = 1.5
```

但注意：**不要为了速度牺牲环境模型质量**！

---

## 📊 实验对比

建议进行对比实验：

### 实验A：无warmup（原始）
```python
warmup_epochs = 0  # 相当于禁用
env_loss_weight = 1.0
```

### 实验B：有warmup（新方案）
```python
warmup_epochs = 10
env_loss_weight = 2.0
```

### 对比指标

训练50 epoch后，对比：
```
            Exp A (无warmup)   Exp B (有warmup)
val_bis_mae      2.8              2.3          ✅
val_rp_mae       5.2              4.1          ✅
val_action_mae   5.5              5.3          ≈
MCTS激活率       15%              25%          ✅
训练稳定性       中等             高           ✅
```

**预期**：
- Exp B的环境模型更准（bis_mae, rp_mae更低）
- MCTS激活率更高（因为模型更准）
- 训练更稳定（loss曲线更平滑）

---

## ⚡ 快速开始

### 1. 确认配置（已设置）

```python
# constant.py
warmup_epochs = 10
env_loss_weight = 2.0
```

### 2. 运行训练

```bash
python train.py
```

### 3. 监控训练

```bash
tensorboard --logdir output/rlditr/log
```

**关注**：
- `in_warmup`：是否在warmup阶段（1.0=是，0.0=否）
- `loss_bis`, `loss_rp`：环境模型loss
- `loss_action`：policy loss（warmup时不优化）
- `loss_mcts`：MCTS激活情况

---

## 🎯 预期效果

### 相对于无warmup

**环境模型质量**：
- `val_bis_mae`改善：10-20%
- `val_rp_mae`改善：10-20%
- 训练稳定性提升

**MCTS效果**：
- MCTS激活率提高（因为模型更准）
- MCTS找到的动作更可靠

**整体性能**：
- `val_action_mae`可能略好或持平
- 但最终BIS/RP控制效果更好

---

## ⚠️ 注意事项

### 1. Warmup不是越长越好

```
warmup太短（<5）    → 环境模型可能不够准
warmup合适（10-20） → 平衡速度和质量
warmup太长（>30）   → 浪费时间，policy收敛慢
```

**推荐**：从10开始，根据`val_bis_mae`调整

### 2. 环境模型权重很重要

```
env_weight太低（=1.0）  → policy可能干扰环境模型
env_weight合适（2.0-3.0）→ 平衡
env_weight太高（>5.0）  → policy收敛很慢
```

**推荐**：从2.0开始

### 3. 医疗场景必须保证环境模型质量

- ❌ 不要为了policy性能牺牲环境模型
- ✅ 环境模型准确 > policy收敛速度
- ✅ 宁可warmup长一点，也要确保模型准

---

## 📚 理论基础

### 为什么warmup有效？

1. **避免早期噪声**
   - 训练初期，所有参数随机
   - 如果同时训练，policy在错误模型上学习
   - Warmup确保模型先稳定

2. **减少model exploitation**
   - 联合训练时，dynamics可能"迎合"policy
   - Warmup确保dynamics先学真实转移
   - 后续即使有干扰，也有坚实基础

3. **类似curriculum learning**
   - 先学简单任务（环境模型）
   - 再学复杂任务（policy优化）
   - 渐进式学习更稳定

### 加权的作用

```python
loss = loss_action + 2.0 * loss_env

# 相当于
loss = loss_action + 2*loss_bis + 2*loss_rp + 2*loss_value + 2*loss_reward

# 梯度反向传播时
grad_dynamics ∝ 2.0  (来自环境loss)
grad_dynamics ∝ 1.0  (来自policy的隐式梯度)

# 环境模型的"真实信号"占主导
```

---

## ✅ 总结

### 实现的改进

1. ✅ 添加warmup机制（前10 epoch只训练环境模型）
2. ✅ 添加加权机制（环境模型权重2倍）
3. ✅ 添加`in_warmup`指标监控
4. ✅ 无需修改其他代码，向后兼容

### 使用建议

```python
# 推荐配置
warmup_epochs = 10
env_loss_weight = 2.0

# 训练时关注
- Epoch 1-10: 环境模型loss下降
- Epoch 11+: 环境loss不反弹，policy开始收敛

# 如果环境模型质量不好
- 延长warmup或增大env_weight
```

### 下一步

1. 运行训练，观察warmup效果
2. 对比有/无warmup的结果
3. 根据环境模型质量调整参数

**环境模型质量是MCTS成功的关键！** 🎯
