"""
MCTS for Anesthesia Drug Control
基于 Transformer 世界模型的蒙特卡洛树搜索
"""

import math
import torch
import numpy as np
from typing import Tuple, Optional
from constant import *


class MCTSNode:
    """MCTS 树节点"""
    
    def __init__(self, state, parent=None, action=None, prior=0.0):
        """
        Args:
            state: 当前状态 (1, L, n_hidden)
            parent: 父节点
            action: 从父节点到当前节点的动作 (action_bbf, action_rftn)
            prior: 先验概率（来自策略网络）
        """
        self.state = state
        self.parent = parent
        self.action = action  # (action_bbf, action_rftn)
        self.prior = prior
        
        self.children = {}  # {(action_bbf, action_rftn): MCTSNode}
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0  # 从父节点转移到当前节点的即时奖励
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        """平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count, c_puct=1.0):
        """UCB 分数：利用 + 探索"""
        # Q(s,a): 平均价值
        q_value = self.value()
        
        # U(s,a): 探索奖励
        u_value = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct=1.0):
        """选择 UCB 分数最高的子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def expand(self, action_priors):
        """
        扩展节点，为所有可能的动作创建子节点（惰性扩展）
        
        Args:
            action_priors: dict of {(action_bbf, action_rftn): prior_prob}
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=None,  # 状态将在第一次访问时计算
                    parent=self,
                    action=action,
                    prior=prior
                )
    
    def update(self, value):
        """反向传播更新"""
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, model, n_simulations=50, c_puct=1.0, gamma=0.9, 
                 num_sampled_actions=10, temperature=1.0):
        """
        Args:
            model: TransformerPlanningModel
            n_simulations: MCTS 模拟次数
            c_puct: UCB 探索常数
            gamma: 折扣因子
            num_sampled_actions: 从策略中采样的动作数量（降低分支因子）
            temperature: 温度参数，控制最终动作选择的随机性
        """
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.num_sampled_actions = num_sampled_actions
        self.temperature = temperature
        
    def search(self, state_0, state_t, action_prev, option_prev, padding_mask=None, offset=0):
        """
        执行 MCTS 搜索
        
        Args:
            state_0: 初始真实观测的状态 (B, L, n_hidden)
            state_t: 当前状态 (B, L, n_hidden)
            action_prev: 历史动作 (2, B, L)
            option_prev: 历史选项 (B, L)
            padding_mask: 填充掩码
            offset: 时间偏移
            
        Returns:
            best_action: (action_bbf, action_rftn) 标量
            mcts_policy: 改进后的策略分布 (用于训练)
        """
        # 创建根节点
        root = MCTSNode(state=state_t)
        
        # 执行 n_simulations 次模拟
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection: 选择到叶子节点
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # 2. Expansion & Evaluation
            # 如果节点状态未计算，先用 dynamics 计算
            if node.state is None and node.parent is not None:
                parent_state = node.parent.state
                action_bbf, action_rftn = node.action
                
                # 构造动作张量
                action_now = self._construct_action(action_bbf, action_rftn, action_prev)
                option_now = option_prev  # 简化：使用相同的 option
                
                # 使用世界模型预测下一状态和奖励
                with torch.no_grad():
                    next_state, reward = self.model.dynamics(
                        parent_state, action_now, option_now, 
                        state_0=state_0, 
                        padding_mask=padding_mask, 
                        offset=offset
                    )
                
                node.state = next_state
                node.reward = reward.item()
            
            # 获取当前节点的策略和价值
            if node.state is not None:
                with torch.no_grad():
                    policy, value = self.model.prediction(
                        torch.reshape(node.state, (node.state.shape[0], -1))
                    )
                    value = value.item()
                    
                    # 如果是新叶子节点，扩展它
                    if node.visit_count == 0:
                        action_priors = self._get_action_priors(policy)
                        node.expand(action_priors)
            else:
                value = 0.0
            
            # 3. Backup: 反向传播价值
            self._backup(search_path, value)
        
        # 根据访问计数选择最佳动作
        best_action, mcts_policy = self._select_action(root)
        
        return best_action, mcts_policy
    
    def _construct_action(self, action_bbf, action_rftn, action_prev):
        """构造动作张量"""
        action_now_bbf = torch.zeros_like(action_prev[0])
        action_now_rftn = torch.zeros_like(action_prev[1])
        action_now_bbf[:, 0] = action_bbf
        action_now_rftn[:, 0] = action_rftn
        return (action_now_bbf, action_now_rftn)
    
    def _get_action_priors(self, policy):
        """
        从策略网络获取动作先验概率
        为了降低分支因子，只采样 top-k 动作
        
        Args:
            policy: (policy_bbf, policy_rftn) 每个形状 (B, n_action)
            
        Returns:
            action_priors: dict of {(action_bbf, action_rftn): prior}
        """
        policy_bbf, policy_rftn = policy
        
        # 转换为概率分布
        prob_bbf = torch.softmax(policy_bbf[0], dim=-1)
        prob_rftn = torch.softmax(policy_rftn[0], dim=-1)
        
        # 采样 top-k 动作
        top_k = self.num_sampled_actions
        top_bbf_probs, top_bbf_actions = torch.topk(prob_bbf, k=min(top_k, len(prob_bbf)))
        top_rftn_probs, top_rftn_actions = torch.topk(prob_rftn, k=min(top_k, len(prob_rftn)))
        
        action_priors = {}
        
        # 组合两种药物的 top-k 动作
        # 为了进一步降低分支因子，可以只考虑部分组合
        for i, (bbf_action, bbf_prob) in enumerate(zip(top_bbf_actions, top_bbf_probs)):
            for j, (rftn_action, rftn_prob) in enumerate(zip(top_rftn_actions, top_rftn_probs)):
                # 联合概率（假设独立）
                joint_prob = bbf_prob.item() * rftn_prob.item()
                action_priors[(bbf_action.item(), rftn_action.item())] = joint_prob
        
        # 归一化
        total = sum(action_priors.values())
        if total > 0:
            action_priors = {k: v / total for k, v in action_priors.items()}
        
        return action_priors
    
    def _backup(self, search_path, value):
        """反向传播价值"""
        for node in reversed(search_path):
            node.update(value)
            # 折扣未来价值
            value = node.reward + self.gamma * value
    
    def _select_action(self, root):
        """
        根据访问计数选择最佳动作
        
        Returns:
            best_action: (action_bbf, action_rftn)
            mcts_policy: 改进后的策略分布（用于训练）
        """
        # 收集访问计数
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        if not visit_counts:
            # 如果没有子节点，返回默认动作
            return (0, 0), None
        
        # 使用温度参数调整分布
        if self.temperature == 0:
            # 贪婪选择
            best_action = max(visit_counts, key=visit_counts.get)
        else:
            # 根据访问计数的幂次分布采样
            actions = list(visit_counts.keys())
            counts = np.array([visit_counts[a] for a in actions])
            
            # 应用温度
            counts = counts ** (1.0 / self.temperature)
            probs = counts / counts.sum()
            
            # 采样
            idx = np.random.choice(len(actions), p=probs)
            best_action = actions[idx]
        
        # 构造改进后的策略（用于训练）
        mcts_policy = {action: count / sum(visit_counts.values()) 
                       for action, count in visit_counts.items()}
        
        return best_action, mcts_policy


class SimplifiedMCTS:
    """
    简化版 MCTS：只扩展单个药物维度，降低复杂度
    适合快速实验
    """
    
    def __init__(self, model, n_simulations=20, c_puct=1.0, gamma=0.9):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
    
    def search_sequential(self, state_0, state_t, action_prev, option_prev, 
                         padding_mask=None, offset=0):
        """
        顺序搜索：先搜索 BBF，再搜索 RFTN
        大幅降低搜索空间
        """
        # 搜索 BBF
        action_bbf = self._search_single_drug(
            state_0, state_t, action_prev, option_prev, 
            drug_idx=0, padding_mask=padding_mask, offset=offset
        )
        
        # 固定 BBF，搜索 RFTN
        action_rftn = self._search_single_drug(
            state_0, state_t, action_prev, option_prev,
            drug_idx=1, fixed_action=action_bbf, 
            padding_mask=padding_mask, offset=offset
        )
        
        return (action_bbf, action_rftn)
    
    def _search_single_drug(self, state_0, state_t, action_prev, option_prev,
                           drug_idx, fixed_action=None, padding_mask=None, offset=0):
        """
        搜索单个药物的最佳剂量
        
        Args:
            drug_idx: 0 for BBF, 1 for RFTN
            fixed_action: 如果搜索 RFTN，需要固定 BBF 的动作
        """
        with torch.no_grad():
            policy, _ = self.model.prediction(
                torch.reshape(state_t, (state_t.shape[0], -1))
            )
            
            # 获取当前药物的策略分布
            current_policy = policy[drug_idx][0]  # (n_action,)
            probs = torch.softmax(current_policy, dim=-1)
            
            # 采样 top-k 动作进行评估
            top_k = min(10, len(probs))
            top_probs, top_actions = torch.topk(probs, k=top_k)
            
            best_value = -float('inf')
            best_action = top_actions[0].item()
            
            # 评估每个候选动作
            for action, prob in zip(top_actions, top_probs):
                # 构造完整动作
                if drug_idx == 0:  # BBF
                    full_action = (action.item(), 0 if fixed_action is None else fixed_action)
                else:  # RFTN
                    full_action = (fixed_action, action.item())
                
                # 使用世界模型模拟
                action_tensor = self._construct_action(full_action, action_prev)
                next_state, reward = self.model.dynamics(
                    state_t, action_tensor, option_prev,
                    state_0=state_0, padding_mask=padding_mask, offset=offset
                )
                
                # 评估价值
                _, value = self.model.prediction(
                    torch.reshape(next_state, (next_state.shape[0], -1))
                )
                
                total_value = reward.item() + self.gamma * value.item()
                
                if total_value > best_value:
                    best_value = total_value
                    best_action = action.item()
        
        return best_action
    
    def _construct_action(self, action, action_prev):
        """构造动作张量"""
        action_bbf, action_rftn = action
        action_now_bbf = torch.zeros_like(action_prev[0])
        action_now_rftn = torch.zeros_like(action_prev[1])
        action_now_bbf[:, 0] = action_bbf
        action_now_rftn[:, 0] = action_rftn
        return (action_now_bbf, action_now_rftn)
