#!/usr/bin/env python3
"""
RL+BC混合训练实现验证脚本

这个脚本用于验证RL+BC实现的正确性，包括：
1. 配置参数检查
2. 模型输出格式验证
3. 损失计算验证
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from constant import (
        use_rl_bc_hybrid, lambda_bc, lambda_rl, teacher_forcing,
        n_action, n_option, n_reward_max, n_value_max, max_lenth, 
        n_bis, n_rp, n_hidden, pre_len
    )
    from RL.Module.model import TransformerPlanningModel
    from RL.Module.pl_model import RLSLModelModule
    print("✅ 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def verify_config():
    """验证配置参数"""
    print("\n" + "="*60)
    print("1. 配置参数验证")
    print("="*60)
    
    print(f"use_rl_bc_hybrid: {use_rl_bc_hybrid}")
    print(f"lambda_bc: {lambda_bc}")
    print(f"lambda_rl: {lambda_rl}")
    print(f"teacher_forcing: {teacher_forcing}")
    
    # 验证权重和为1
    if abs((lambda_bc + lambda_rl) - 1.0) < 1e-6:
        print("✅ BC和RL权重和为1.0")
    else:
        print(f"⚠️  BC和RL权重和为 {lambda_bc + lambda_rl}，不等于1.0")
    
    # 验证权重范围
    if 0 <= lambda_bc <= 1 and 0 <= lambda_rl <= 1:
        print("✅ 权重在有效范围内 [0, 1]")
    else:
        print("❌ 权重超出有效范围")
        return False
    
    return True


def verify_model_output():
    """验证模型输出格式"""
    print("\n" + "="*60)
    print("2. 模型输出格式验证")
    print("="*60)
    
    try:
        # 创建模型
        n_aux = 3
        n_input = 381
        model = TransformerPlanningModel(
            n_action=n_action,
            n_option=n_option,
            n_reward_max=n_reward_max,
            n_value_max=n_value_max,
            max_lenth=max_lenth,
            n_aux=n_aux,
            n_input=n_input,
        )
        print("✅ 模型创建成功")
        
        # 创建虚拟输入
        batch_size = 2
        obs = torch.randn(batch_size, max_lenth, n_input)
        action_prev = (
            torch.randint(0, n_action, (batch_size, max_lenth + pre_len)),
            torch.randint(0, n_action, (batch_size, max_lenth + pre_len))
        )
        option_prev = torch.randint(0, n_option, (batch_size, max_lenth + pre_len))
        padding_mask = torch.zeros(batch_size, max_lenth, dtype=torch.bool)
        
        print(f"✅ 输入数据创建成功")
        print(f"   - obs shape: {obs.shape}")
        print(f"   - action_prev shapes: {action_prev[0].shape}, {action_prev[1].shape}")
        print(f"   - option_prev shape: {option_prev.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(obs, action_prev, option_prev, padding_mask, gamma=0.9)
        
        print("✅ 前向传播成功")
        
        # 验证输出
        required_keys = [
            'policy', 'value', 'reward', 'bis', 'rp',
            'policy_train', 'policy_action_logprob',
            'policy_reward', 'policy_return',
        ]
        
        if use_rl_bc_hybrid:
            required_keys.extend(['rl_action_target', 'rl_weights'])
        
        missing_keys = [k for k in required_keys if k not in outputs]
        if missing_keys:
            print(f"❌ 缺少输出键: {missing_keys}")
            return False
        
        print("✅ 所有必需的输出键都存在")
        
        # 验证RL+BC特定输出
        if use_rl_bc_hybrid:
            rl_action_target = outputs['rl_action_target']
            rl_weights = outputs['rl_weights']
            
            print(f"   - rl_action_target shapes: {rl_action_target[0].shape}, {rl_action_target[1].shape}")
            print(f"   - rl_weights shape: {rl_weights.shape}")
            
            # 验证形状
            expected_shape = (batch_size, pre_len)
            if rl_action_target[0].shape == expected_shape and rl_action_target[1].shape == expected_shape:
                print("✅ RL目标动作形状正确")
            else:
                print(f"❌ RL目标动作形状错误，期望 {expected_shape}")
                return False
            
            if rl_weights.shape == expected_shape:
                print("✅ RL权重形状正确")
            else:
                print(f"❌ RL权重形状错误，期望 {expected_shape}")
                return False
            
            # 验证权重范围
            if (rl_weights >= 0).all() and (rl_weights <= 1).all():
                print("✅ RL权重在有效范围 [0, 1]")
            else:
                print("❌ RL权重超出有效范围")
                return False
            
            # 统计RL触发率
            rl_trigger_rate = rl_weights.mean().item()
            print(f"ℹ️  RL触发率: {rl_trigger_rate*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_loss_computation():
    """验证损失计算"""
    print("\n" + "="*60)
    print("3. 损失计算验证")
    print("="*60)
    
    try:
        from RL.Module.model import TransformerPlanningModel
        from RL.Module.pl_model import RLSLModelModule
        
        # 创建模型
        n_aux = 3
        n_input = 381
        model = TransformerPlanningModel(
            n_action=n_action,
            n_option=n_option,
            n_reward_max=n_reward_max,
            n_value_max=n_value_max,
            max_lenth=max_lenth,
            n_aux=n_aux,
            n_input=n_input,
        )
        
        # 创建训练模块
        pl_model = RLSLModelModule(
            model=model,
            lr=0.001,
            model_args={},
            module_args={'loss_rl_joint': True}
        )
        
        print("✅ 训练模块创建成功")
        
        # 创建虚拟数据
        batch_size = 2
        obs = torch.randn(batch_size, max_lenth, n_input)
        action_all = (
            torch.randint(0, n_action, (batch_size, max_lenth + pre_len)),
            torch.randint(0, n_action, (batch_size, max_lenth + pre_len))
        )
        option_all = torch.randint(0, n_option, (batch_size, max_lenth + pre_len))
        padding = torch.zeros(batch_size, max_lenth, dtype=torch.bool)
        
        # 创建标签数据
        label_data = {
            'action': (
                action_all[0][:, max_lenth:max_lenth+pre_len],
                action_all[1][:, max_lenth:max_lenth+pre_len]
            ),
            'mask_reward': torch.ones(batch_size, max_lenth + pre_len),
            'bis_target': torch.randint(0, n_bis+1, (batch_size, max_lenth + pre_len)),
            'rp_target': torch.randint(0, n_rp+1, (batch_size, max_lenth + pre_len)),
            'reward': torch.randn(batch_size, max_lenth + pre_len),
            'cumreward': torch.randn(batch_size, max_lenth + pre_len),
            'padding': padding,
        }
        
        # 前向传播
        with torch.no_grad():
            pred_data = model(obs, action_all, option_all, padding, gamma=0.9)
        
        # Stack label data
        label_data_stacked, stack_mask = pl_model.label_data_stack(label_data, pre_len)
        
        # 计算损失
        losses = pl_model.get_loss(pred_data, label_data_stacked)
        
        print("✅ 损失计算成功")
        
        # 验证损失键
        required_loss_keys = ['loss', 'loss_action', 'loss_bc', 'loss_rl']
        missing_loss_keys = [k for k in required_loss_keys if k not in losses]
        if missing_loss_keys:
            print(f"❌ 缺少损失键: {missing_loss_keys}")
            return False
        
        print("✅ 所有必需的损失键都存在")
        
        # 打印损失值
        print(f"\n损失值:")
        print(f"   - loss_action: {losses['loss_action'].item():.4f}")
        print(f"   - loss_bc: {losses['loss_bc'].item():.4f}")
        print(f"   - loss_rl: {losses['loss_rl'].item():.4f}")
        
        # 验证混合损失
        if use_rl_bc_hybrid:
            # 验证损失组合
            expected_loss = lambda_bc * losses['loss_bc'] + lambda_rl * losses['loss_rl']
            actual_loss = losses['loss_action']
            
            # 由于可能有数值误差，使用相对误差
            if torch.allclose(expected_loss, actual_loss, rtol=1e-5):
                print("✅ 混合损失计算正确")
                print(f"   期望: {expected_loss.item():.4f}")
                print(f"   实际: {actual_loss.item():.4f}")
            else:
                print("❌ 混合损失计算错误")
                print(f"   期望: {expected_loss.item():.4f}")
                print(f"   实际: {actual_loss.item():.4f}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 损失计算验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*60)
    print("RL+BC混合训练实现验证")
    print("="*60)
    
    results = []
    
    # 1. 配置验证
    results.append(("配置参数", verify_config()))
    
    # 2. 模型输出验证
    results.append(("模型输出", verify_model_output()))
    
    # 3. 损失计算验证
    results.append(("损失计算", verify_loss_computation()))
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有验证通过！RL+BC实现正确。")
        print("\n建议下一步：")
        print("1. 运行baseline实验（设置 use_rl_bc_hybrid=False）")
        print("2. 运行RL+BC实验（保守配置：lambda_bc=0.7）")
        print("3. 对比性能指标")
    else:
        print("⚠️  部分验证失败，请检查实现。")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
