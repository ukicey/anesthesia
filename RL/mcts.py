"""
MCTS（蒙特卡洛树搜索）模块

用于在训练Stage 2时搜索高价值动作，引导policy学习
基于MuZero的思路，使用学到的环境模型（dynamics, reward, value）进行搜索
"""

import torch
import math
import numpy as np
from typing import Tuple, Optional


class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, prior_bbf: float, prior_rftn: float, parent=None):
        """
        Args:
            prior_bbf: BBF动作的先验概率
            prior_rftn: RFTN动作的先验概率
            parent: 父节点
        """
        self.parent = parent
        self.prior_bbf = prior_bbf
        self.prior_rftn = prior_rftn
        
        # 访问统计
        self.visit_count = 0
        self.value_sum = 0.0
        
        # 子节点: {(action_bbf, action_rftn): MCTSNode}
        self.children = {}
        
        # 当前状态
        self.reward = 0.0
        self.state = None  # hidden state
    
    def value(self) -> float:
        """节点的平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_expanded(self) -> bool:
        """是否已扩展"""
        return len(self.children) > 0
    
    def expand(self, policy_bbf, policy_rftn, state, reward):
        """
        扩展节点
        
        Args:
            policy_bbf: BBF动作的策略分布 (n_action,)
            policy_rftn: RFTN动作的策略分布 (n_action,)
            state: 当前隐状态
            reward: 到达此节点的奖励
        """
        self.state = state
        self.reward = reward
        
        # 转换为概率分布
        policy_bbf = torch.softmax(policy_bbf, dim=-1)
        policy_rftn = torch.softmax(policy_rftn, dim=-1)
        
        # 为了效率，只扩展top-k动作
        # 在医疗场景中，可以扩展所有动作或top-k
        # 这里简化：扩展每个动作独立的top-k组合
        k_bbf = min(10, policy_bbf.shape[-1])  # top-10 BBF动作
        k_rftn = min(10, policy_rftn.shape[-1])  # top-10 RFTN动作
        
        top_bbf_probs, top_bbf_actions = torch.topk(policy_bbf, k_bbf)
        top_rftn_probs, top_rftn_actions = torch.topk(policy_rftn, k_rftn)
        
        # 创建子节点（组合top-k动作）
        for i, (a_bbf, p_bbf) in enumerate(zip(top_bbf_actions, top_bbf_probs)):
            for j, (a_rftn, p_rftn) in enumerate(zip(top_rftn_actions, top_rftn_probs)):
                action = (int(a_bbf), int(a_rftn))
                prior = float(p_bbf * p_rftn)  # 联合概率
                self.children[action] = MCTSNode(prior_bbf=float(p_bbf), 
                                                 prior_rftn=float(p_rftn), 
                                                 parent=self)
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[Tuple[int, int], 'MCTSNode']:
        """
        选择子节点（UCB算法）
        
        Args:
            c_puct: 探索常数
            
        Returns:
            (action, child_node)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # UCB分数 = Q值 + U值（探索奖励）
            q_value = child.value()
            
            # 探索奖励: c_puct * P(a) * sqrt(N_parent) / (1 + N_child)
            u_value = c_puct * (child.prior_bbf * child.prior_rftn) * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value: float):
        """反向传播更新节点价值"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            self.parent.update(value)


class MCTS:
    """
    MCTS搜索器
    
    使用学到的模型（representation, dynamics, prediction）进行搜索
    """
    
    def __init__(self, model, n_simulations=50, c_puct=1.0, gamma=0.9):
        """
        Args:
            model: TransformerPlanningModel
            n_simulations: 每次搜索的模拟次数
            c_puct: UCB探索常数
            gamma: 折扣因子
        """
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
    
    @torch.no_grad()
    def search(self, 
               state_0: torch.Tensor,
               state_t: torch.Tensor,
               action_prev: Tuple[torch.Tensor, torch.Tensor],
               option_t: torch.Tensor,
               padding_mask: Optional[torch.Tensor] = None,
               offset: int = 0) -> Tuple[int, int]:
        """
        执行MCTS搜索，返回最优动作
        
        Args:
            state_0: 初始真实观测状态 (B, L, n_hidden)
            state_t: 当前状态 (B, L, n_hidden)
            action_prev: 之前的动作模板 (B, L)
            option_t: 当前option (B, L)
            padding_mask: 填充掩码
            offset: 时间偏移
            
        Returns:
            (action_bbf, action_rftn): 最优动作
        """
        # 创建根节点
        policy_t, value_t = self.model.prediction(torch.reshape(state_t, (state_t.shape[0], -1)))
        root = MCTSNode(prior_bbf=1.0, prior_rftn=1.0, parent=None)
        root.expand(policy_t[0][0], policy_t[1][0], state_t, reward=0.0)
        
        # 执行n次模拟
        for _ in range(self.n_simulations):
            self._simulate(root, state_0, action_prev, option_t, padding_mask, offset)
        
        # 选择访问次数最多的动作
        best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        return best_action
    
    def _simulate(self, node: MCTSNode, state_0, action_prev, option_t, padding_mask, offset):
        """
        执行一次MCTS模拟
        
        1. Selection: 从根节点开始，选择到叶子节点
        2. Expansion: 扩展叶子节点
        3. Evaluation: 评估叶子节点
        4. Backpropagation: 反向传播价值
        """
        # 记录路径
        path = [node]
        
        # 1. Selection - 选择到叶子节点
        current_state = node.state
        while node.is_expanded():
            action, node = node.select_child(self.c_puct)
            path.append(node)
            
            # 如果是第一次访问，需要用dynamics推进
            if node.state is None:
                # 构造动作张量
                action_tensor = self._construct_action_tensor(action, action_prev)
                # 用dynamics推进
                next_state, reward = self.model.dynamics(
                    current_state, action_tensor, option_t,
                    state_0=state_0, padding_mask=padding_mask, offset=offset
                )
                
                # 获取下一状态的policy和value
                policy_next, value_next = self.model.prediction(
                    torch.reshape(next_state, (next_state.shape[0], -1))
                )
                
                # 2. Expansion - 扩展节点
                node.expand(policy_next[0][0], policy_next[1][0], next_state, reward[0].item())
                
                # 3. Evaluation - 使用value网络评估
                leaf_value = value_next[0].item()
                
                # 3. Backpropagation - 反向传播
                # 价值 = 累积折扣奖励 + 折扣的叶子价值
                value = leaf_value
                for i in range(len(path) - 1, -1, -1):
                    value = path[i].reward + self.gamma * value
                    path[i].update(value)
                
                return
            
            current_state = node.state
        
        # 如果到达未扩展的叶子，评估并扩展
        policy_t, value_t = self.model.prediction(torch.reshape(current_state, (current_state.shape[0], -1)))
        node.expand(policy_t[0][0], policy_t[1][0], current_state, reward=0.0)
        
        # 反向传播
        value = value_t[0].item()
        for i in range(len(path) - 1, -1, -1):
            value = path[i].reward + self.gamma * value
            path[i].update(value)
    
    def _construct_action_tensor(self, action, action_prev):
        """构造动作张量"""
        action_bbf, action_rftn = action
        action_now_bbf = torch.zeros_like(action_prev[0])
        action_now_rftn = torch.zeros_like(action_prev[1])
        action_now_bbf[:, 0] = action_bbf
        action_now_rftn[:, 0] = action_rftn
        return (action_now_bbf, action_now_rftn)
    
    def get_action_probs(self, 
                         state_0: torch.Tensor,
                         state_t: torch.Tensor,
                         action_prev: Tuple[torch.Tensor, torch.Tensor],
                         option_t: torch.Tensor,
                         temperature: float = 1.0,
                         padding_mask: Optional[torch.Tensor] = None,
                         offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行MCTS搜索，返回改进的策略分布（而不是单一动作）
        
        这个可以用于生成MCTS改进的目标策略，让policy学习
        
        Args:
            temperature: 温度参数，控制探索程度
                - temperature=0: 选择访问最多的动作（确定性）
                - temperature=1: 按访问次数比例采样
                
        Returns:
            (improved_policy_bbf, improved_policy_rftn): 改进的策略分布
        """
        # 创建根节点并搜索
        policy_t, value_t = self.model.prediction(torch.reshape(state_t, (state_t.shape[0], -1)))
        root = MCTSNode(prior_bbf=1.0, prior_rftn=1.0, parent=None)
        root.expand(policy_t[0][0], policy_t[1][0], state_t, reward=0.0)
        
        # 执行搜索
        for _ in range(self.n_simulations):
            self._simulate(root, state_0, action_prev, option_t, padding_mask, offset)
        
        # 收集访问计数
        n_action_bbf = policy_t[0].shape[-1]
        n_action_rftn = policy_t[1].shape[-1]
        
        visit_counts_bbf = torch.zeros(n_action_bbf, device=policy_t[0].device)
        visit_counts_rftn = torch.zeros(n_action_rftn, device=policy_t[1].device)
        
        for (action_bbf, action_rftn), child in root.children.items():
            visit_counts_bbf[action_bbf] += child.visit_count
            visit_counts_rftn[action_rftn] += child.visit_count
        
        # 应用温度
        if temperature == 0:
            # 确定性：选择访问最多的
            improved_policy_bbf = torch.zeros_like(visit_counts_bbf)
            improved_policy_rftn = torch.zeros_like(visit_counts_rftn)
            improved_policy_bbf[visit_counts_bbf.argmax()] = 1.0
            improved_policy_rftn[visit_counts_rftn.argmax()] = 1.0
        else:
            # 按温度调整的访问计数
            visit_counts_bbf = visit_counts_bbf ** (1.0 / temperature)
            visit_counts_rftn = visit_counts_rftn ** (1.0 / temperature)
            
            # 归一化为概率分布
            improved_policy_bbf = visit_counts_bbf / (visit_counts_bbf.sum() + 1e-8)
            improved_policy_rftn = visit_counts_rftn / (visit_counts_rftn.sum() + 1e-8)
        
        return improved_policy_bbf, improved_policy_rftn
