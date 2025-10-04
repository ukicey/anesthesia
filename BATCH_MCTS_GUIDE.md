# 批量MCTS使用指南

## ✅ 完成的实现

现在MCTS**支持批量操作**！

### 之前的问题
```python
# 旧版本：只对第1个样本搜索
action_mcts = mcts.search(state[0:1], ...)  # 只处理batch[0]
# batch[1], batch[2], ... batch[127]都没用MCTS ❌
```

### 现在的解决方案
```python
# 新版本：对所有样本搜索
actions_mcts = mcts.batch_search(state, ...)  # 处理整个batch ✅
# 返回 (actions_bbf, actions_rftn)，每个都是 (B,) tensor
```

---

## 📊 三种模式

### 模式1：快速模式（推荐）⚡

```python
# constant.py
use_mcts_batch_search = False  # 快速模式
mcts_batch_samples = 32        # 每batch只对32个样本MCTS
```

**工作方式**：
- 对batch中前32个样本执行MCTS
- 其余样本使用贪婪策略
- 平衡速度和效果

**适用场景**：
- ✅ 训练阶段（batch_size=128）
- ✅ 资源有限
- ✅ 需要快速迭代

**速度**：
- batch_size=128，只搜索32个
- 相比旧版（只搜索1个），覆盖率提升32倍
- 相比完整模式，速度快4倍

### 模式2：完整模式（最准确）🎯

```python
# constant.py
use_mcts_batch_search = True  # 完整模式
# mcts_batch_samples 无效（会搜索所有）
```

**工作方式**：
- 对batch中**所有样本**执行MCTS
- 每个样本都得到MCTS优化的动作

**适用场景**：
- ✅ 验证/测试阶段
- ✅ 资源充足
- ✅ 需要最佳效果

**速度**：
- batch_size=128，搜索所有128个
- 最慢，但效果最好

### 模式3：禁用模式（baseline）🚫

```python
# constant.py
use_mcts = False
```

**工作方式**：
- 完全不使用MCTS
- 纯BC（行为克隆）

**适用场景**：
- ✅ Baseline对比
- ✅ 调试环境模型

---

## ⚙️ 配置参数

### 推荐配置（训练）

```python
# constant.py

# 启用快速批量MCTS
use_mcts = True
use_mcts_batch_search = False  # 快速模式
mcts_batch_samples = 32        # 每batch搜索32个样本

# MCTS搜索参数
mcts_simulations = 50    # 每个样本50次模拟
mcts_c_puct = 1.0
use_mcts_margin = 0.1

# BC+MCTS混合
lambda_bc = 0.7
lambda_mcts = 0.3
teacher_forcing = True
```

**预期**：
- MCTS覆盖率：32/128 = 25%
- 训练时间：约1.5-2倍baseline
- MCTS激活率：5-15%（因为只覆盖25%样本）

### 完整配置（验证/测试）

```python
# 完整批量MCTS
use_mcts = True
use_mcts_batch_search = True   # 完整模式
mcts_simulations = 80          # 增加模拟次数

# 其他参数同上
```

**预期**：
- MCTS覆盖率：100%
- 训练时间：约5-10倍baseline
- MCTS激活率：10-40%（完整覆盖）

### 调优建议

#### 如果训练太慢

```python
# 方案1：减少搜索样本
mcts_batch_samples = 16  # 或8

# 方案2：减少模拟次数
mcts_simulations = 30  # 或20

# 方案3：禁用MCTS（临时）
use_mcts = False
```

#### 如果想要更好效果

```python
# 方案1：增加搜索样本
mcts_batch_samples = 64  # 或batch_size//2

# 方案2：使用完整模式
use_mcts_batch_search = True

# 方案3：增加模拟次数
mcts_simulations = 80
```

---

## 📈 性能对比

### 覆盖率

| 模式 | MCTS样本数 | 覆盖率 | 相对旧版 |
|------|-----------|--------|---------|
| 旧版 | 1 | 0.78% | 1x |
| 快速 (32) | 32 | 25% | **32x** ✅ |
| 快速 (64) | 64 | 50% | **64x** ✅✅ |
| 完整 | 128 | 100% | **128x** ✅✅✅ |

### 训练时间

假设baseline（无MCTS）= 1x

| 模式 | 训练时间 | 吞吐量 |
|------|---------|--------|
| Baseline | 1x | 100% |
| 旧版 (1样本) | 1.1x | 91% |
| 快速 (32样本) | 1.5-2x | 50-67% |
| 快速 (64样本) | 2-3x | 33-50% |
| 完整 (128样本) | 5-10x | 10-20% |

### MCTS激活率

假设单样本MCTS激活率=30%

| 模式 | 整体激活率 | 说明 |
|------|-----------|------|
| 旧版 | 0.23% | 1/128 × 30% |
| 快速 (32) | 7.5% | 32/128 × 30% |
| 快速 (64) | 15% | 64/128 × 30% |
| 完整 | 30% | 128/128 × 30% |

---

## 🔍 实现细节

### batch_search（完整模式）

```python
def batch_search(self, state, ...):
    """对batch中所有样本执行MCTS"""
    batch_size = state.shape[0]
    actions_bbf = []
    actions_rftn = []
    
    for b in range(batch_size):
        # 对每个样本独立搜索
        action_bbf, action_rftn = self.search(state[b:b+1], ...)
        actions_bbf.append(action_bbf)
        actions_rftn.append(action_rftn)
    
    return torch.tensor(actions_bbf), torch.tensor(actions_rftn)
```

**特点**：
- 简单循环，每个样本独立
- 不需要复杂的并行协调
- 稳定可靠

### batch_search_fast（快速模式）

```python
def batch_search_fast(self, state, ..., max_samples=32):
    """只对部分样本执行MCTS"""
    # 先用贪婪策略初始化所有动作
    policy = model.prediction(state)
    actions_bbf = policy[0].argmax(dim=-1)  # (B,)
    actions_rftn = policy[1].argmax(dim=-1)
    
    # 对前max_samples个样本用MCTS覆盖
    for b in range(min(max_samples, batch_size)):
        action_bbf, action_rftn = self.search(state[b:b+1], ...)
        actions_bbf[b] = action_bbf
        actions_rftn[b] = action_rftn
    
    return actions_bbf, actions_rftn
```

**特点**：
- 部分样本MCTS，其余贪婪
- 平衡速度和效果
- 灵活可调（max_samples）

---

## 📊 监控指标

### 新增指标

训练时查看：

```python
# MCTS覆盖统计
mcts_coverage = mcts_weight.sum() / batch_size
# 快速模式(32)：约25%
# 完整模式：约100%

# MCTS激活率（真正找到更优动作）
mcts_activation = (mcts_weight > 0).float().mean()
# 理想：10-40%（在被搜索的样本中）
```

### 训练日志示例

```
# 快速模式
Epoch 15:
  loss_bc: 1.234
  loss_mcts: 0.089  ✅ > 0
  mcts_coverage: 25%
  mcts_activation: 12%  (3.75%整体 = 25% × 15%)

# 完整模式  
Epoch 15:
  loss_bc: 1.234
  loss_mcts: 0.245  ✅ 更高
  mcts_coverage: 100%
  mcts_activation: 30%  (30%整体)
```

---

## 🎯 使用建议

### 训练流程

#### Phase 1: Warm-up（Epoch 1-10）
```python
use_mcts = False  # 暂时禁用
# 只训练环境模型
```

#### Phase 2: 快速MCTS（Epoch 11-50）
```python
use_mcts = True
use_mcts_batch_search = False
mcts_batch_samples = 32
mcts_simulations = 50
```

#### Phase 3: 完整MCTS（Epoch 51-100，可选）
```python
use_mcts_batch_search = True
mcts_simulations = 80
```

### 对比实验

建议进行消融实验：

| 实验 | 配置 | 目的 |
|------|------|------|
| A | use_mcts=False | Baseline |
| B | mcts_batch_samples=1 | 旧版MCTS |
| C | mcts_batch_samples=32 | 快速批量 |
| D | use_mcts_batch_search=True | 完整批量 |

**对比指标**：
- `val_bis_mae`（主要）
- `val_action_mae`
- 训练时间
- MCTS激活率

---

## ⚠️ 注意事项

### 1. 内存消耗

完整模式会展开batch_size个MCTS树：

```
内存消耗 ≈ batch_size × mcts_simulations × tree_size

# 示例
batch_size=128, simulations=50
→ 6400个节点需要存储
→ 可能需要10-20GB GPU内存
```

**解决方案**：
- 使用快速模式
- 减小batch_size
- 减少mcts_simulations

### 2. 训练时间

完整批量MCTS很慢：

```
# 单step时间估算
baseline: 0.1s
快速(32): 0.2s  (2x)
完整(128): 1.0s (10x)

# 完整epoch时间
1000 steps/epoch:
  baseline: 100s
  快速: 200s
  完整: 1000s (17分钟)
```

**建议**：
- 训练用快速模式
- 验证/测试用完整模式

### 3. 批量大小影响

MCTS效果依赖batch_size：

```
batch_size=16, mcts_samples=8 → 50%覆盖 ✅
batch_size=256, mcts_samples=32 → 12.5%覆盖 ⚠️
```

**建议**：
- 保持`mcts_samples / batch_size ≥ 0.25`
- 或使用完整模式

---

## ✅ 总结

### 实现的改进

1. ✅ **批量接口**：`batch_search()`和`batch_search_fast()`
2. ✅ **灵活配置**：完整模式vs快速模式
3. ✅ **性能提升**：覆盖率提升32-128倍
4. ✅ **向后兼容**：可以禁用回到baseline

### 推荐配置

**训练（推荐）**：
```python
use_mcts = True
use_mcts_batch_search = False
mcts_batch_samples = 32
mcts_simulations = 50
```

**验证/测试**：
```python
use_mcts = True
use_mcts_batch_search = True
mcts_simulations = 80
```

### 预期效果

相比旧版（只搜索1个样本）：
- ✅ MCTS覆盖率提升32倍
- ✅ 整体MCTS激活率从0.23%→7.5%
- ⚠️ 训练时间增加1.5-2倍（可接受）

相比纯BC：
- ✅ 性能提升5-15%（如果环境模型准确）
- ⚠️ 训练时间增加1.5-2倍

---

## 🚀 开始使用

### 1. 确认配置（已设置好推荐值）

```python
# constant.py
use_mcts = True
use_mcts_batch_search = False  # 快速模式
mcts_batch_samples = 32
```

### 2. 运行训练

```bash
python train.py
```

### 3. 监控训练

```bash
tensorboard --logdir output/rlditr/log
```

查看：
- `loss_mcts > 0`：MCTS在工作
- 对比有/无MCTS的性能

---

**现在MCTS可以充分发挥作用了！** 🎉
