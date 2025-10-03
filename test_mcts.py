#!/usr/bin/env python3
"""
测试 MCTS 模块是否正常工作
"""

import torch
import numpy as np
from RL.Module.model import TransformerPlanningModel
from constant import *

def test_mcts_basic():
    """测试 MCTS 基本功能"""
    print("=" * 60)
    print("测试 MCTS 基本功能")
    print("=" * 60)
    
    # 创建一个小模型用于测试
    model = TransformerPlanningModel(
        n_action=n_action,
        n_option=n_option,
        n_reward_max=n_reward_max,
        n_value_max=n_value_max,
        max_lenth=max_lenth,
        n_aux=3,
        n_input=100,  # 简化
        use_mcts=True,
        mcts_simulations=5  # 少量模拟用于快速测试
    )
    model.eval()
    
    # 创建随机输入
    batch_size = 2
    obs = torch.randn(batch_size, max_lenth, 100)
    action_prev = (
        torch.randint(0, n_action, (batch_size, max_lenth)),
        torch.randint(0, n_action, (batch_size, max_lenth))
    )
    option = torch.randint(0, n_option, (batch_size, max_lenth))
    padding_mask = torch.zeros(batch_size, max_lenth, dtype=torch.bool)
    
    print(f"输入形状:")
    print(f"  obs: {obs.shape}")
    print(f"  action_prev: ({action_prev[0].shape}, {action_prev[1].shape})")
    print(f"  option: {option.shape}")
    
    # 测试初始推理
    print("\n测试初始推理...")
    with torch.no_grad():
        policy, state_0 = model.initial_inference(obs, action_prev, option, padding_mask)
    
    print(f"  state_0: {state_0.shape}")
    print(f"  policy_bbf: {policy[0].shape}, policy_rftn: {policy[1].shape}")
    
    # 测试 MCTS 搜索
    print("\n测试 MCTS 搜索...")
    if model.use_mcts:
        state_t = state_0
        action_mcts = model.mcts.search_sequential(
            state_0[0:1], state_t[0:1], 
            (action_prev[0][0:1], action_prev[1][0:1]),
            option[0:1], padding_mask=padding_mask[0:1], offset=0
        )
        print(f"  MCTS 选择的动作: BBF={action_mcts[0]}, RFTN={action_mcts[1]}")
        print(f"  ✅ MCTS 搜索成功！")
    else:
        print("  ❌ MCTS 未启用")
    
    # 测试贪婪选择（对比）
    print("\n测试贪婪选择（对比）...")
    greedy_action = (
        policy[0][0].argmax().item(),
        policy[1][0].argmax().item()
    )
    print(f"  贪婪选择的动作: BBF={greedy_action[0]}, RFTN={greedy_action[1]}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


def test_mcts_forward():
    """测试完整的 forward 过程"""
    print("\n" + "=" * 60)
    print("测试完整 forward 过程（带 MCTS）")
    print("=" * 60)
    
    model = TransformerPlanningModel(
        n_action=n_action,
        n_option=n_option,
        n_reward_max=n_reward_max,
        n_value_max=n_value_max,
        max_lenth=max_lenth,
        n_aux=3,
        n_input=100,
        use_mcts=True,
        mcts_simulations=3  # 极少模拟，只测试流程
    )
    model.eval()
    
    batch_size = 1
    total_len = max_lenth + pre_len
    
    obs = torch.randn(batch_size, max_lenth, 100)
    action_prev = (
        torch.randint(0, n_action, (batch_size, total_len)),
        torch.randint(0, n_action, (batch_size, total_len))
    )
    option = torch.randint(0, n_option, (batch_size, total_len))
    padding_mask = torch.zeros(batch_size, max_lenth, dtype=torch.bool)
    
    print("执行 forward（这可能需要一些时间）...")
    
    try:
        with torch.no_grad():
            outputs = model(obs, action_prev, option, padding_mask, gamma=0.9)
        
        print(f"\n输出形状:")
        print(f"  policy_bbf: {outputs['policy'][0].shape}")
        print(f"  policy_rftn: {outputs['policy'][1].shape}")
        print(f"  value: {outputs['value'].shape}")
        print(f"  reward: {outputs['reward'].shape}")
        print(f"  bis: {outputs['bis'].shape}")
        print(f"  rp: {outputs['rp'].shape}")
        
        print("\n✅ Forward 测试通过！MCTS 在 Stage 2 正常工作")
        
    except Exception as e:
        print(f"\n❌ Forward 测试失败: {e}")
        raise
    
    print("=" * 60)


def compare_with_without_mcts():
    """对比有无 MCTS 的差异"""
    print("\n" + "=" * 60)
    print("对比有无 MCTS 的差异")
    print("=" * 60)
    
    # 准备相同的输入
    batch_size = 1
    total_len = max_lenth + pre_len
    
    torch.manual_seed(42)  # 固定随机种子
    obs = torch.randn(batch_size, max_lenth, 100)
    action_prev = (
        torch.randint(0, n_action, (batch_size, total_len)),
        torch.randint(0, n_action, (batch_size, total_len))
    )
    option = torch.randint(0, n_option, (batch_size, total_len))
    padding_mask = torch.zeros(batch_size, max_lenth, dtype=torch.bool)
    
    # 模型 1: 不使用 MCTS
    print("\n1️⃣ 不使用 MCTS（贪婪选择）")
    model_no_mcts = TransformerPlanningModel(
        n_action=n_action, n_option=n_option, n_reward_max=n_reward_max,
        n_value_max=n_value_max, max_lenth=max_lenth, n_aux=3, n_input=100,
        use_mcts=False
    )
    model_no_mcts.eval()
    
    import time
    start = time.time()
    with torch.no_grad():
        outputs_no_mcts = model_no_mcts(obs, action_prev, option, padding_mask)
    time_no_mcts = time.time() - start
    
    print(f"  执行时间: {time_no_mcts:.3f}s")
    print(f"  平均奖励: {outputs_no_mcts['reward'].mean().item():.4f}")
    
    # 模型 2: 使用 MCTS
    print("\n2️⃣ 使用 MCTS（树搜索）")
    model_with_mcts = TransformerPlanningModel(
        n_action=n_action, n_option=n_option, n_reward_max=n_reward_max,
        n_value_max=n_value_max, max_lenth=max_lenth, n_aux=3, n_input=100,
        use_mcts=True, mcts_simulations=5
    )
    
    # 复制权重确保公平对比
    model_with_mcts.load_state_dict(model_no_mcts.state_dict(), strict=False)
    model_with_mcts.eval()
    
    start = time.time()
    with torch.no_grad():
        outputs_with_mcts = model_with_mcts(obs, action_prev, option, padding_mask)
    time_with_mcts = time.time() - start
    
    print(f"  执行时间: {time_with_mcts:.3f}s")
    print(f"  平均奖励: {outputs_with_mcts['reward'].mean().item():.4f}")
    
    # 对比
    print("\n📊 对比结果:")
    print(f"  时间开销: {time_with_mcts / time_no_mcts:.2f}x")
    print(f"  奖励差异: {outputs_with_mcts['reward'].mean().item() - outputs_no_mcts['reward'].mean().item():.4f}")
    
    print("\n💡 说明:")
    print("  - MCTS 会增加计算时间（正常现象）")
    print("  - 奖励差异可能很小（因为模型未训练）")
    print("  - 训练后，MCTS 通常能获得更高奖励")
    
    print("=" * 60)


if __name__ == '__main__':
    print("\n🧪 MCTS 模块测试\n")
    
    try:
        # 测试 1: 基本功能
        test_mcts_basic()
        
        # 测试 2: Forward 过程
        test_mcts_forward()
        
        # 测试 3: 对比实验
        compare_with_without_mcts()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！MCTS 模块工作正常")
        print("=" * 60)
        print("\n💡 下一步:")
        print("  1. 在 constant.py 中设置 use_mcts=True")
        print("  2. 运行 python train.py train 开始训练")
        print("  3. 监控训练指标（loss、MAE）的改善")
        print("  4. 调整 mcts_simulations 平衡速度和性能")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("  1. 是否正确安装了所有依赖")
        print("  2. constant.py 中的参数是否正确")
        print("  3. MCTS 模块是否正确导入")
