# MCTS改善Stage 2训练 - 完整指南

## 🎯 核心思路

### 之前的问题
你之前的反思文档（`MCTS_CORRECT_UNDERSTANDING.md`）正确地指出了问题：

**问题**：在纯BC框架下，MCTS只改变rollout路径，但policy的loss仍然只和专家动作比较，没有学习MCTS找到的好动作。

```python
# ❌ 错误做法（之前）
for i in range(pre_len):
    policy_t = model.prediction(state_t)
    
    if use_mcts:
        action = mcts.search(state_t)  # MCTS选择动作
    else:
        action = policy_t.argmax()  # 贪婪选择
    
    state_t_next = model.dynamics(state_t, action, ...)  # 用action推进
    
    # 但loss仍然只和专家比较！
    loss = CrossEntropy(policy_t, action_expert)  # ← 问题在这里
```

**结果**：MCTS只影响rollout轨迹，policy永远在学习专家，无法利用MCTS找到的高价值动作。

### 现在的解决方案

**核心**：让policy**同时学习**专家动作（BC）和MCTS找到的高价值动作（MCTS引导）

```python
# ✅ 正确做法（现在）
for i in range(pre_len):
    policy_t = model.prediction(state_t)
    action_expert = action_label[i]  # 专家动作
    
    if use_mcts:
        # MCTS搜索找到高价值动作
        action_mcts = mcts.search(state_t)
        
        # 评估：MCTS动作 vs 专家动作
        Q_mcts = evaluate(state_t, action_mcts)
        Q_expert = evaluate(state_t, action_expert)
        
        # 如果MCTS找到更好的动作，学习它
        if Q_mcts > Q_expert + margin:
            mcts_target = action_mcts
            mcts_weight = 1.0
        else:
            mcts_target = action_expert
            mcts_weight = 0.0
    
    # BC loss：始终学习专家
    loss_bc = CE(policy_t, action_expert)
    
    # MCTS loss：有条件地学习MCTS找到的好动作
    loss_mcts = CE(policy_t, mcts_target) * mcts_weight
    
    # 混合损失
    loss = λ_bc * loss_bc + λ_mcts * loss_mcts
    
    # Rollout：推荐用专家动作（稳定）
    state_t_next = model.dynamics(state_t, action_expert, ...)
```

---

## 🏗️ 实现架构

### 1. MCTS模块（`RL/mcts.py`）

基于MuZero的思路，使用学到的环境模型进行搜索：

```
MCTS搜索树
├── 使用学到的模型组件
│   ├── prediction: 预测policy和value
│   ├── dynamics: 预测next_state和reward
│   └── representation: 初始状态编码
├── UCB算法选择节点
│   └── score = Q(s,a) + c * P(a) * sqrt(N_parent) / (1 + N_child)
└── 搜索n次模拟后返回最优动作
```

**关键特性**：
- 完全在学到的模型内搜索（无需真实环境）
- 利用dynamics和reward预测评估动作
- 平衡探索（未访问的动作）与利用（高价值动作）

### 2. Model修改（`RL/Module/model.py`）

**Stage 2的新流程**：

```python
for i in range(pre_len):
    # 1. Prediction
    policy_t, value_t = model.prediction(state_t)
    action_expert = action_label[i]
    
    # 2. MCTS搜索（训练时）
    if use_mcts and training:
        action_mcts = mcts.search(state_0, state_t, ...)
        
        # 3. 评估两个动作
        Q_mcts = reward_mcts + γ * value_mcts
        Q_expert = reward_expert + γ * value_expert
        
        # 4. 选择MCTS目标
        if Q_mcts > Q_expert + margin:
            mcts_target = action_mcts  # 学习MCTS
            mcts_weight = 1.0
        else:
            mcts_target = action_expert  # 回退到专家
            mcts_weight = 0.0
    
    # 5. Rollout（推荐用专家，稳定）
    if teacher_forcing:
        state_t = dynamics(state_t, action_expert, ...)
    else:
        state_t = dynamics(state_t, mcts_target, ...)
    
    # 6. 输出mcts_target和mcts_weight用于计算loss
```

**新增输出**：
- `mcts_action_target`: MCTS找到的目标动作
- `mcts_weights`: MCTS的置信权重（1.0=找到更好动作，0.0=用专家）

### 3. Loss计算（`RL/Module/pl_model.py`）

```python
# BC损失：永远学习专家（安全性保证）
loss_bc = CE(policy, action_expert)

# MCTS损失：学习MCTS找到的高价值动作
if use_mcts:
    loss_mcts = CE(policy, mcts_target) * mcts_weights
else:
    loss_mcts = 0

# 混合损失
loss_action = λ_bc * loss_bc + λ_mcts * loss_mcts
```

**权重配置**（`constant.py`）：
```python
lambda_bc = 0.7    # BC权重（推荐0.6-0.8）
lambda_mcts = 0.3  # MCTS权重（推荐0.2-0.4）
```

---

## ⚙️ 配置参数

### 基础配置（`constant.py`）

```python
# ==== 核心开关 ====
use_mcts = True  # 是否使用MCTS
teacher_forcing = True  # 是否用专家动作rollout（推荐True）

# ==== 损失权重 ====
lambda_bc = 0.7    # BC权重：保证安全性
lambda_mcts = 0.3  # MCTS权重：允许优化

# ==== MCTS搜索参数 ====
mcts_simulations = 50  # 每次搜索的模拟次数（越大越慢但越准）
mcts_c_puct = 1.0      # UCB探索常数（越大越探索）
use_mcts_margin = 0.1  # MCTS动作需要比专家好多少（margin）
```

### 推荐配置方案

#### 🥇 方案A：保守（医疗推荐）
```python
use_mcts = True
lambda_bc = 0.8
lambda_mcts = 0.2
teacher_forcing = True
mcts_simulations = 30
use_mcts_margin = 0.15
```
**特点**：最大化安全性，MCTS需要显著更好才使用

#### 🥈 方案B：平衡
```python
use_mcts = True
lambda_bc = 0.7
lambda_mcts = 0.3
teacher_forcing = True
mcts_simulations = 50
use_mcts_margin = 0.1
```
**特点**：平衡安全性和优化能力（推荐起点）

#### 🥉 方案C：激进（实验）
```python
use_mcts = True
lambda_bc = 0.5
lambda_mcts = 0.5
teacher_forcing = False
mcts_simulations = 80
use_mcts_margin = 0.05
```
**特点**：最大化优化潜力，可能不稳定

#### Baseline：纯BC
```python
use_mcts = False
# 其他参数无关
```
**特点**：原始方案，用于对比

---

## 🚀 使用方法

### Step 1: 确认配置

编辑 `constant.py`:
```python
use_mcts = True
lambda_bc = 0.7
lambda_mcts = 0.3
teacher_forcing = True
mcts_simulations = 50
```

### Step 2: 修改train.py（如果需要）

确保模型创建时传入MCTS参数：

```python
# train.py 或 创建模型的地方
from constant import use_mcts, mcts_simulations, mcts_c_puct

model = TransformerPlanningModel(
    n_action=n_action,
    n_option=n_option,
    n_reward_max=n_reward_max,
    n_value_max=n_value_max,
    max_lenth=max_lenth,
    n_aux=n_aux,
    n_input=n_input,
    # MCTS参数
    use_mcts=use_mcts,
    mcts_simulations=mcts_simulations,
    mcts_c_puct=mcts_c_puct,
)
```

### Step 3: 运行训练

```bash
# 先跑baseline（纯BC）
# 在constant.py中设置 use_mcts = False
python train.py

# 再跑BC+MCTS
# 在constant.py中设置 use_mcts = True
python train.py
```

### Step 4: 监控指标

训练时关注：
- `loss_bc`: BC损失（应稳定下降）
- `loss_mcts`: MCTS损失（可能波动，表示MCTS激活频率）
- `loss_action`: 总损失（主要优化目标）
- `val_action_mae`: 验证集动作误差

**健康训练的标志**：
- `loss_bc` 稳定下降
- `loss_mcts` 保持在合理范围（不应过大）
- `val_action_mae` ≤ baseline

---

## 📊 MCTS如何改善训练

### 理论优势

1. **超越专家的能力**
   - BC只能学习专家水平
   - MCTS可以找到比专家更优的动作
   - 混合方案：保持专家基线 + 允许改进

2. **利用学到的环境模型**
   - MCTS在学到的dynamics/reward上搜索
   - 类似"在脑海中模拟"，找到高价值路径
   - 不需要额外的真实数据

3. **自举式改进**
   - 随着模型变好，MCTS搜索也变好
   - 更好的MCTS引导更好的policy
   - 正反馈循环

4. **安全性保证**
   - λ_bc = 0.7确保始终学习专家
   - MCTS只在"确信"更好时才激活
   - Margin机制避免微小差异的噪声

### 实际效果

**预期提升**（相对baseline）:
- 保守配置（λ_bc=0.8）：+3-10%
- 平衡配置（λ_bc=0.7）：+5-15%
- 激进配置（λ_bc=0.5）：+10-25%（或失败）

**MCTS激活率**：
- 理想范围：10-40%
- 太低(<5%)：MCTS很少找到更好动作，可能：
  - 专家已经很好
  - Margin太大
  - MCTS搜索次数太少
- 太高(>60%)：可能过于乐观，检查：
  - Margin是否太小
  - 环境模型是否准确

---

## 🔍 工作原理详解

### MCTS搜索过程

```
1. Selection（选择）
   从根节点开始，用UCB算法选择到叶子节点
   UCB = Q(s,a) + c * P(a) * sqrt(N_parent) / (1 + N_child)

2. Expansion（扩展）
   扩展叶子节点，创建子节点
   使用prediction函数获取prior probabilities

3. Simulation（模拟）
   使用dynamics函数推进环境
   使用value函数评估叶子状态

4. Backpropagation（反向传播）
   更新路径上所有节点的访问次数和价值
   value = reward_t + γ * value_{t+1}

重复n次后，选择访问次数最多的动作
```

### 价值评估

```python
# 评估动作的Q值
Q(s, a) = R(s, a) + γ * V(s')

其中：
- R(s, a): reward函数预测的即时奖励
- V(s'): value函数预测的未来回报
- γ: 折扣因子（0.9）
```

### 动作选择逻辑

```python
if Q_mcts > Q_expert + margin:
    # MCTS找到显著更好的动作
    target = action_mcts
    weight = 1.0  # 施加MCTS loss
else:
    # 专家动作足够好或MCTS未找到更优
    target = action_expert
    weight = 0.0  # 只施加BC loss
```

### Teacher Forcing

```python
# Rollout动作选择
if teacher_forcing:
    # 用专家动作推进dynamics（推荐）
    # 优点：轨迹稳定，不会偏离专家分布
    # 缺点：无法探索MCTS找到的新路径
    state_next = dynamics(state, action_expert, ...)
else:
    # 用MCTS动作推进dynamics（激进）
    # 优点：完全探索MCTS路径
    # 缺点：可能偏离专家分布，训练不稳定
    state_next = dynamics(state, mcts_target, ...)
```

---

## 🐛 故障排查

### 问题1: MCTS loss始终为0

**现象**：`loss_mcts = 0.0`持续不变

**原因**：MCTS从未找到比专家更好的动作

**解决方案**：
1. 降低margin：`use_mcts_margin = 0.05`或`0.0`
2. 增加搜索次数：`mcts_simulations = 80`或`100`
3. 检查环境模型质量：
   ```bash
   # 查看reward和value的loss
   # 如果reward_loss和value_loss很大，先训练好环境模型
   ```
4. 可能专家已经很好，这是正常的

### 问题2: 训练速度变慢

**现象**：训练时间增加2-5倍

**原因**：MCTS搜索计算量大

**解决方案**：
1. 减少搜索次数：`mcts_simulations = 30`或`20`
2. 只在部分step使用MCTS（修改代码）
3. 使用GPU加速（确保所有计算在GPU上）
4. 减少batch size以平衡内存和速度

### 问题3: 性能下降

**现象**：BC+MCTS不如纯BC

**原因**：
- 环境模型不准确，MCTS学到错误策略
- MCTS权重过高，偏离专家太多
- Margin太小，学到噪声而非真实改进

**解决方案**：
1. 增加BC权重：`lambda_bc = 0.8, lambda_mcts = 0.2`
2. 增大margin：`use_mcts_margin = 0.2`
3. 启用teacher forcing：`teacher_forcing = True`
4. 检查环境模型loss（reward, value, bis, rp）
5. 如果环境模型不好，先不用MCTS

### 问题4: 训练不稳定

**现象**：loss曲线剧烈波动

**原因**：
- Rollout用MCTS动作，偏离专家分布
- MCTS搜索结果不一致
- 权重设置不当

**解决方案**：
1. 启用teacher forcing：`teacher_forcing = True`
2. 降低学习率：`lr = 0.00005`
3. 增加batch size：`batch_size = 256`
4. 增大margin：`use_mcts_margin = 0.15`

### 问题5: GPU内存不足

**现象**：`CUDA out of memory`

**原因**：MCTS搜索展开大量节点

**解决方案**：
1. 减少batch size
2. 减少MCTS搜索次数
3. 在MCTS搜索中只展开top-k动作（已实现）
4. 使用CPU进行MCTS（修改代码）

---

## 📈 实验建议

### 实验流程

1. **Baseline（纯BC）**
   ```python
   use_mcts = False
   ```
   记录：`val_action_mae`, `val_bis_mae`, `val_rp_mae`

2. **BC+MCTS（保守）**
   ```python
   use_mcts = True
   lambda_bc = 0.8
   lambda_mcts = 0.2
   mcts_simulations = 30
   ```
   对比：是否至少不低于baseline

3. **BC+MCTS（平衡）**
   ```python
   lambda_bc = 0.7
   lambda_mcts = 0.3
   mcts_simulations = 50
   ```
   对比：是否有进一步提升

4. **参数调优**
   - 调整λ_bc/λ_mcts比例
   - 调整margin
   - 调整搜索次数

### 监控清单

- [ ] `loss_bc`稳定下降
- [ ] `loss_mcts`在合理范围（0-loss_bc之间）
- [ ] `val_action_mae` ≤ baseline
- [ ] MCTS激活率在10-40%
- [ ] 训练时间可接受（<baseline的3倍）
- [ ] GPU内存充足

### 成功指标

✅ **成功**：
- `val_action_mae`降低5-15%
- 训练稳定，loss平滑下降
- MCTS适度激活（10-40%）

⚠️ **需调整**：
- 性能提升<3%：增加λ_mcts，降低margin
- 性能下降：增加λ_bc，增大margin
- 训练不稳定：启用teacher forcing，增大margin

❌ **失败**：
- 性能持续下降：暂停MCTS，检查环境模型
- 训练崩溃：降低搜索次数，减小batch size

---

## 💡 关键洞察

### 1. MCTS不是万能的

MCTS只在以下情况有效：
- ✅ 环境模型（dynamics, reward, value）训练良好
- ✅ 专家数据充足但可能次优
- ✅ 有计算资源进行搜索
- ❌ 环境模型不准确→MCTS在错误模型上优化
- ❌ 专家已经最优→MCTS无法改进
- ❌ 资源有限→搜索开销太大

### 2. BC是安全网

- λ_bc = 0.7确保始终学习专家
- 即使MCTS完全失败，模型仍有70%专家水平
- 医疗场景：安全性 > 性能

### 3. Teacher Forcing的权衡

**Teacher Forcing = True（推荐）**：
- 优点：训练稳定，不偏离专家分布
- 缺点：无法完全探索MCTS路径
- 适合：医疗、生产环境

**Teacher Forcing = False（激进）**：
- 优点：完全探索MCTS，可能更优
- 缺点：可能偏离分布，训练不稳定
- 适合：研究、实验

### 4. MCTS vs 简单RL

你可能会问：为什么不直接用简单的RL（如我之前实现的）？

**MCTS的优势**：
- 🔍 **更深入的搜索**：MCTS展开搜索树，考虑多步后果
- 🎯 **更准确的评估**：通过多次模拟评估动作，而非单次前向
- 🌳 **平衡探索与利用**：UCB算法自动平衡
- 🏆 **AlphaZero证明**：在围棋等领域超越人类

**简单RL的优势**：
- ⚡ **更快**：无需MCTS搜索
- 💾 **更省内存**：不展开搜索树
- 🔧 **更简单**：实现和调试更容易

**建议**：
- 如果计算资源充足，用MCTS
- 如果资源有限，用简单RL
- 可以两者都试，对比效果

---

## 📚 理论基础

这个方案结合了多个经典思路：

1. **AlphaZero/MuZero**
   - 使用学到的模型进行MCTS搜索
   - Policy学习MCTS搜索结果（policy improvement）
   - Self-play式的自举改进

2. **Behavior Cloning（BC）**
   - 模仿专家提供安全基线
   - 经典的模仿学习方法

3. **Expert Iteration**
   - 专家（MCTS）改进策略
   - 策略模仿专家
   - 迭代提升

4. **Offline RL with BC regularization**
   - BC提供正则化，避免偏离数据分布
   - 允许策略改进，但有约束

---

## 🎓 进阶话题

### 动态调整权重

可以在训练过程中动态调整λ_bc和λ_mcts：

```python
# 训练初期：高BC权重（稳定）
if epoch < 20:
    lambda_bc = 0.9
    lambda_mcts = 0.1
# 训练中期：平衡
elif epoch < 50:
    lambda_bc = 0.7
    lambda_mcts = 0.3
# 训练后期：高MCTS权重（优化）
else:
    lambda_bc = 0.5
    lambda_mcts = 0.5
```

### MCTS策略分布

当前实现返回单一最优动作，也可以返回改进的策略分布：

```python
# 使用get_action_probs而非search
improved_policy = mcts.get_action_probs(state, temperature=1.0)

# 用KL散度让policy学习改进的分布
loss_mcts = KL(policy_pred, improved_policy)
```

### 并行化MCTS

当前实现只对batch中第一个样本搜索，可以并行化：

```python
# 对batch中每个样本并行MCTS
mcts_actions = parallel_mcts_search(states_batch)
```

---

## ✅ 检查清单

使用MCTS前：
- [ ] 环境模型（dynamics, reward, value）训练良好
- [ ] 有baseline（纯BC）性能记录
- [ ] GPU内存足够（至少8GB）
- [ ] 理解MCTS原理和参数含义
- [ ] 设置了合理的λ_bc, λ_mcts, margin

训练时：
- [ ] 监控loss_bc和loss_mcts
- [ ] 监控MCTS激活率
- [ ] 对比baseline性能
- [ ] 检查训练稳定性

出问题时：
- [ ] 增加BC权重
- [ ] 增大margin
- [ ] 启用teacher forcing
- [ ] 减少搜索次数
- [ ] 检查环境模型质量

---

## 🚀 开始使用！

最简单的开始方式：

```bash
# 1. 确认配置（constant.py已设置）
use_mcts = True
lambda_bc = 0.7
lambda_mcts = 0.3
teacher_forcing = True

# 2. 运行训练
python train.py

# 3. 监控指标
tensorboard --logdir output/rlditr/log
```

**记住**：
- 🎯 MCTS是改进而非替代BC
- 🛡️ BC权重70%保证安全性
- 🔍 MCTS搜索找到更优动作
- 📊 监控指标，及时调整

祝训练成功！🎉

---

## 📞 参考资料

- 你的反思文档中的"方案3：混合BC+RL"就是这个思路
- AlphaZero论文：https://arxiv.org/abs/1712.01815
- MuZero论文：https://arxiv.org/abs/1911.08265
- Expert Iteration: https://arxiv.org/abs/1705.08439
