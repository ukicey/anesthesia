# 混合 BC + RL 的 Stage 2 实现示例
# 修改 model.py 的 forward 函数中的 Stage 2 部分

def forward_stage2_hybrid(self, state_0, action_prev, option_prev, action_真实, padding_mask):
    """
    Stage 2: 混合 BC (Behavior Cloning) 和 RL
    
    BC: 模仿真实动作（安全、稳定）
    RL: 利用 MCTS 搜索的高价值动作（优化性能）
    """
    state_t = state_0
    option_t = option_prev
    
    policy_list_bbf = []
    policy_list_rftn = []
    bc_loss_list = []
    rl_loss_list = []
    
    λ_bc = 0.7  # BC 权重
    λ_rl = 0.3  # RL 权重
    
    for i in range(pre_len):
        # 1. 预测策略
        policy_t, value_t = self.prediction(torch.reshape(state_t, (state_t.shape[0], -1)))
        
        dist_bbf = torch.distributions.Categorical(logits=policy_t[0])
        dist_rftn = torch.distributions.Categorical(logits=policy_t[1])
        
        # 2. 获取真实动作
        action_真实_bbf = action_真实[0][:, i]
        action_真实_rftn = action_真实[1][:, i]
        
        # 3. BC 损失（模仿真实动作）
        loss_bc_bbf = F.cross_entropy(policy_t[0], action_真实_bbf)
        loss_bc_rftn = F.cross_entropy(policy_t[1], action_真实_rftn)
        loss_bc = loss_bc_bbf + loss_bc_rftn
        
        # 4. RL 损失（如果使用 MCTS）
        if self.use_mcts:
            # MCTS 搜索
            action_mcts = self.mcts.search_sequential(
                state_0, state_t, action_prev, option_t,
                padding_mask=padding_mask, offset=1+i
            )
            
            # 评估 MCTS 动作的价值
            action_mcts_tensor = self._construct_action(action_mcts, action_prev)
            next_state_mcts, reward_mcts = self.dynamics(
                state_t, action_mcts_tensor, option_t, 
                state_0=state_0, padding_mask=padding_mask, offset=1+i
            )
            _, value_mcts = self.prediction(torch.reshape(next_state_mcts, (next_state_mcts.shape[0], -1)))
            Q_mcts = reward_mcts + gamma * value_mcts
            
            # 评估真实动作的价值
            action_真实_tensor = self._construct_action(
                (action_真实_bbf[0].item(), action_真实_rftn[0].item()), 
                action_prev
            )
            next_state_真实, reward_真实 = self.dynamics(
                state_t, action_真实_tensor, option_t,
                state_0=state_0, padding_mask=padding_mask, offset=1+i
            )
            _, value_真实 = self.prediction(torch.reshape(next_state_真实, (next_state_真实.shape[0], -1)))
            Q_真实 = reward_真实 + gamma * value_真实
            
            # 如果 MCTS 找到更好的动作，学习它
            if Q_mcts > Q_真实:
                action_mcts_bbf = torch.tensor([action_mcts[0]], device=state_t.device)
                action_mcts_rftn = torch.tensor([action_mcts[1]], device=state_t.device)
                
                loss_rl_bbf = F.cross_entropy(policy_t[0], action_mcts_bbf)
                loss_rl_rftn = F.cross_entropy(policy_t[1], action_mcts_rftn)
                loss_rl = loss_rl_bbf + loss_rl_rftn
                
                # 用 MCTS 动作推进
                action_for_rollout = action_mcts_tensor
            else:
                loss_rl = 0
                action_for_rollout = action_真实_tensor
        else:
            loss_rl = 0
            action_for_rollout = self._construct_action(
                (action_真实_bbf[0].item(), action_真实_rftn[0].item()),
                action_prev
            )
        
        # 5. 混合损失
        loss_total = λ_bc * loss_bc + λ_rl * loss_rl if isinstance(loss_rl, torch.Tensor) else loss_bc
        
        # 6. 推进状态
        state_t_next, reward_t = self.dynamics(
            state_t, action_for_rollout, option_t,
            state_0=state_0, padding_mask=padding_mask, offset=1+i
        )
        
        state_t = state_t_next
        
        policy_list_bbf.append(policy_t[0])
        policy_list_rftn.append(policy_t[1])
        bc_loss_list.append(loss_bc)
        rl_loss_list.append(loss_rl if isinstance(loss_rl, torch.Tensor) else torch.tensor(0.0))
    
    return {
        'policy': (torch.stack(policy_list_bbf, dim=1), torch.stack(policy_list_rftn, dim=1)),
        'bc_loss': torch.stack(bc_loss_list).mean(),
        'rl_loss': torch.stack(rl_loss_list).mean(),
    }

def _construct_action(self, action, action_prev):
    """辅助函数：构造动作张量"""
    action_bbf, action_rftn = action
    action_now_bbf = torch.zeros_like(action_prev[0])
    action_now_rftn = torch.zeros_like(action_prev[1])
    action_now_bbf[:, 0] = action_bbf
    action_now_rftn[:, 0] = action_rftn
    return (action_now_bbf, action_now_rftn)
