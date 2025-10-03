# RL+BC混合训练快速开始指南

## 📋 前置检查

### 1. 确认环境依赖
```bash
# 确保已安装所需包
pip install torch pytorch-lightning torchmetrics
```

### 2. 运行验证脚本
```bash
cd /workspace
python3 verify_rl_bc.py
```

**预期输出**：
```
============================================================
RL+BC混合训练实现验证
============================================================
✅ 导入成功

============================================================
1. 配置参数验证
============================================================
use_rl_bc_hybrid: True
lambda_bc: 0.7
lambda_rl: 0.3
teacher_forcing: True
✅ BC和RL权重和为1.0
✅ 权重在有效范围内 [0, 1]

[... 更多验证输出 ...]

🎉 所有验证通过！RL+BC实现正确。
```

---

## 🚀 三步开始训练

### Step 1: 配置参数（`constant.py`）

**推荐配置（保守）**：
```python
# RL+BC混合训练配置
use_rl_bc_hybrid = True   # 启用RL+BC混合
lambda_bc = 0.7           # BC权重70%（安全性优先）
lambda_rl = 0.3           # RL权重30%（允许优化）
teacher_forcing = True    # 使用专家动作rollout（稳定）
```

### Step 2: 运行baseline（纯BC）

为了对比效果，先运行baseline：
```python
# constant.py - 临时修改
use_rl_bc_hybrid = False
```

```bash
python train.py
```

**记录指标**：
- `val_loss`
- `val_action_mae`
- `val_bis_mae`
- `val_rp_mae`

### Step 3: 运行RL+BC混合训练

恢复RL+BC配置：
```python
# constant.py
use_rl_bc_hybrid = True
lambda_bc = 0.7
lambda_rl = 0.3
teacher_forcing = True
```

```bash
python train.py
```

**监控新增指标**：
- `val_loss_bc`: BC损失（应该和baseline相近）
- `val_loss_rl`: RL损失（表示找到更优动作的频率）
- `val_loss_action`: 总损失（主要优化目标）

---

## 📊 如何判断是否成功

### ✅ 成功的标志

1. **训练稳定**：
   - `loss_bc` 平滑下降，类似纯BC
   - `loss_rl` 保持在合理范围（不应远大于`loss_bc`）

2. **性能提升**：
   - `val_action_mae` ≤ baseline
   - `val_bis_mae` ≤ baseline
   - `val_rp_mae` ≤ baseline

3. **RL激活率适中**：
   - 查看验证脚本输出的"RL触发率"
   - 理想范围：5-30%
   - 太低(<1%)：模型很少找到更优动作
   - 太高(>50%)：可能环境模型过于乐观

### ⚠️ 需要调整的情况

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 性能下降 | RL学到错误策略 | 增加`lambda_bc`到0.8，启用`teacher_forcing` |
| 训练不稳定 | Rollout偏离太大 | 启用`teacher_forcing=True` |
| 没有改进 | 专家数据已是最优 | 这是正常的，BC已经足够好 |
| RL loss过大 | Margin阈值太低 | 在`model.py`中增大margin到0.2或0.3 |

---

## 🔧 常用配置方案

### 方案A: 保守（推荐用于医疗场景）
```python
use_rl_bc_hybrid = True
lambda_bc = 0.8
lambda_rl = 0.2
teacher_forcing = True
```
**特点**: 最大化安全性，适度优化

### 方案B: 平衡
```python
use_rl_bc_hybrid = True
lambda_bc = 0.6
lambda_rl = 0.4
teacher_forcing = True
```
**特点**: 平衡安全性和性能

### 方案C: 激进（仅用于实验）
```python
use_rl_bc_hybrid = True
lambda_bc = 0.5
lambda_rl = 0.5
teacher_forcing = False
```
**特点**: 最大化优化潜力，可能不稳定

### 方案D: 纯BC（baseline）
```python
use_rl_bc_hybrid = False
```
**特点**: 原始方案，用于对比

---

## 📈 实验流程

### 第一轮：验证稳定性
1. 运行baseline（纯BC）- 记录性能
2. 运行保守配置（方案A）- 对比性能
3. 确认：RL+BC至少不低于baseline

### 第二轮：寻找最佳配置
1. 尝试方案B（平衡）
2. 如果稳定且有提升，尝试方案C
3. 选择性能最好且稳定的配置

### 第三轮：细调参数
1. 调整`lambda_bc/lambda_rl`比例
2. 尝试不同的`teacher_forcing`设置
3. 调整margin阈值（在`model.py`的`margin=0.1`）

---

## 🐛 故障排查

### 问题1: ImportError
```bash
ModuleNotFoundError: No module named 'torch'
```
**解决**: 安装依赖
```bash
pip install -r requirements.txt
```

### 问题2: 验证脚本失败
**检查清单**:
- [ ] PyTorch版本 >= 1.8
- [ ] constant.py中的配置参数已更新
- [ ] 没有语法错误（运行`python3 -m py_compile RL/Module/model.py`）

### 问题3: 训练崩溃
**可能原因**:
1. GPU内存不足 → 减小`batch_size`
2. 梯度爆炸 → 降低学习率
3. 数据问题 → 检查数据加载

### 问题4: 性能不如baseline
**调试步骤**:
1. 检查`loss_rl`是否异常大
2. 查看RL触发率是否过高
3. 尝试增加`lambda_bc`
4. 启用`teacher_forcing`
5. 如果仍然不行，暂时禁用RL+BC

---

## 📊 监控指标详解

### 训练阶段
```
Epoch 10: 100%|████████| 100/100 [00:30<00:00,  3.33it/s, 
  loss=1.234,           # 总loss（所有组件）
  train_loss=1.234]     # 训练loss
```

### 验证阶段
```
val_loss: 1.123          # 验证总loss
val_loss_bc: 0.856       # BC损失（应稳定下降）
val_loss_rl: 0.267       # RL损失（可能波动）
val_loss_action: 0.778   # 动作总损失（主要指标）
val_action_mae: 5.234    # 动作预测误差（越小越好）
val_bis_mae: 2.345       # BIS预测误差
val_rp_mae: 3.456        # 呼吸频率预测误差
```

### 关键指标优先级
1. **val_action_mae** - 最重要，直接反映策略质量
2. **val_bis_mae, val_rp_mae** - 生命体征预测准确性
3. **val_loss_bc** - BC学习效果
4. **val_loss_rl** - RL优化效果

---

## 💡 最佳实践

### DO ✅
1. **先跑baseline** - 总是先验证纯BC性能
2. **保守开始** - 从高BC权重（0.7-0.8）开始
3. **监控趋势** - 关注loss曲线，不只看最终值
4. **多次实验** - 至少运行3次取平均
5. **记录配置** - 详细记录每次实验的超参数

### DON'T ❌
1. **不要盲目激进** - 不要一开始就用50-50权重
2. **不要忽视baseline** - 没有对比就无法判断改进
3. **不要过度调参** - 如果baseline已经很好，不需要强求改进
4. **不要忽视安全性** - 医疗场景安全性 > 性能
5. **不要急于下结论** - 至少观察20-50个epoch

---

## 🎯 预期结果

### 典型性能提升（相对baseline）

**保守配置**（λ_bc=0.7-0.8）:
- Action MAE: +3-8%
- BIS MAE: +0-5%
- RP MAE: +0-5%
- 训练时间: +5-10%

**平衡配置**（λ_bc=0.5-0.6）:
- Action MAE: +5-15%
- BIS MAE: +3-10%
- RP MAE: +3-10%
- 训练时间: +10-20%

**激进配置**（λ_bc<0.5）:
- 可能+10-20%，也可能性能下降
- 不推荐用于生产环境

---

## 📝 检查清单

启动训练前，确认：

- [ ] 运行了`verify_rl_bc.py`且全部通过
- [ ] 已跑了baseline（纯BC）并记录了性能
- [ ] `constant.py`中配置正确
  - [ ] `use_rl_bc_hybrid`设置正确
  - [ ] `lambda_bc + lambda_rl = 1.0`
  - [ ] 选择了合适的`teacher_forcing`模式
- [ ] 数据路径配置正确
- [ ] 有足够的磁盘空间保存checkpoint
- [ ] GPU内存足够（或已调整batch_size）

---

## 🚀 开始训练！

```bash
# 1. 验证实现
python3 verify_rl_bc.py

# 2. 运行训练
python train.py

# 3. 监控训练（另一个终端）
tensorboard --logdir output/rlditr/log
```

**祝你训练顺利！如有问题，参考：**
- `RL_BC_HYBRID_APPROACH.md` - 详细理论
- `IMPLEMENTATION_SUMMARY.md` - 实现细节
- `verify_rl_bc.py` - 验证脚本

---

## 📞 支持

如果遇到问题：
1. 查看上述文档
2. 运行验证脚本诊断
3. 检查loss曲线是否异常
4. 尝试降低BC权重或启用teacher forcing

记住：**安全第一，性能第二**。在医疗场景中，稳定可靠的BC方案也是好方案！
