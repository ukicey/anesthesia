from typing import Optional

import torch
import torch.nn as nn

from RL.baseline import TransformerCausalEncoder, MLP, TransformerCausalDecoder
from constant import *

# 懒加载MCTS，避免循环导入
_mcts_module = None

def get_mcts():
    global _mcts_module
    if _mcts_module is None:
        from RL.mcts import MCTS
        _mcts_module = MCTS
    return _mcts_module


class TransformerPlanningModel(nn.Module):
    """
    A model with (1) representation, (2) prediction, (3) dynamics functions
    Paper: https://www.nature.com/articles/s41591-023-02552-9
    """
    def __init__(self, n_action, n_option, n_reward_max, n_value_max, max_lenth, n_aux, n_input,
                 use_mcts=False, mcts_simulations=50, mcts_c_puct=1.0):
        super().__init__()
        self.model_type = 'Transformer'
        self.n_action = n_action  # 201
        self.n_option = n_option  # 2，用一个0/1表示
        self.n_reward_max = n_reward_max
        self.n_value_max = n_value_max
        self.n_step = max_lenth
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.nhead = nhead
        self.nlayers = nlayers
        self.nhid = nhid
        self.dropout = dropout
        self.n_reward_size = n_reward_max * 2 + 1
        self.n_value_size = n_value_max * 2 + 1
        self.n_aux = n_aux
        
        # MCTS配置
        self.use_mcts = use_mcts
        self.mcts = None
        if use_mcts:
            MCTS = get_mcts()
            self.mcts = MCTS(
                model=self,
                n_simulations=mcts_simulations,
                c_puct=mcts_c_puct,
                gamma=gamma
            )

        # 输入：state | action | action | option
        # 其中 action 做了嵌入，维度为 n_hidden（必要性存疑，因为 action 本身就是离散有序的）
        self.encoder = TransformerCausalEncoder(n_input + n_hidden * 2 + 1, n_hidden, nhead, nhid, nlayers, max_len=max_lenth, dropout=dropout)

        # 组装MLP层，建模动作空间里的概率分布情况（两种药物的剂量），单隐层，输入为拉直了的过去时间段的 states
        self.output_policy_bbf = MLP(n_hidden*max_lenth, [n_hidden], n_action)
        self.output_policy_rftn = MLP(n_hidden*max_lenth, [n_hidden], n_action)

        # reward 和 dynamics 的输入是 state 和 两个actions，其余函数都是 state
        self.output_value_func = MLP(n_hidden * max_lenth, [n_hidden], 1)  # 输出标量 value
        self.output_reward_func = MLP((n_hidden * 3 + 1) * max_lenth, [n_hidden], 1)  # 输出标量 reward
        self.embedding_action = nn.Embedding(n_action, n_hidden)  # 将离散的整数索引映射为嵌入
        self.output_bis_func = MLP(n_hidden * max_lenth, [n_hidden], n_bis+1)  # 只有一个值
        self.output_rp_func = MLP(n_hidden * max_lenth, [n_hidden], n_rp+1)  # 只有一个呼吸频率
        self.dynamics_func = TransformerCausalDecoder(n_hidden * 3 + 1, n_hidden, nhead, nhid, nlayers, max_len=max_lenth, dropout=dropout)

    def ext_feat(self, obs, action, option):
        """
        从 obs、action、option 提取嵌入 feature

        Args:
            obs: 观测序列 (B, L, n_input)
            action (tuple): 一对动作序列 2 * (B, L)
            option: (B, L)

        Returns:
            feature: (B, L, n_input + n_hidden * 2 + 1)
        """
        action_embed = (self.embedding_action(action[0]), self.embedding_action(action[1]))
        option_ = option.unsqueeze(-1)  # option的最后一维原本是标量，新增一个1维的维度
        feature = torch.cat([obs, action_embed[0], action_embed[1], option_], dim=-1)
        return feature  # (B, L, n_input + n_hidden * 2 + 1) 或 (B, L, n_hidden*3+1)

    @torch.jit.export
    def representation(self, obs, action_prev, option, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        """
        生成初始真实观测序列的 state_0 向量

        Args:
            obs: 观测序列 (B, L, n_input)
            action_prev (tuple): 一对动作序列 2 * (B, L)
            option: (B, L)
            padding_mask: 填充掩码
            dynamics: 动态构造掩码开关

        Returns:
            state_0: (B, L, n_hidden)
        """
        # 只在初始推理时传obs给extra函数，后面都是传state
        feature = self.ext_feat(obs, action_prev, option)  # (B, L, n_input + n_hidden * 2 + 1)
        state_0 = self.encoder(feature, padding_mask, dynamics)
        return state_0  # (B, L, n_hidden)

    @torch.jit.export
    def prediction(self, state: torch.Tensor):
        """
        根据当前 state 预测策略和价值

        Args:
            state: (B, n_hidden * max_lenth)

        Returns:
            (policy_bbf, policy_rftn): 两种药物的策略分布 (B, n_action)
            value: 价值 (B, 1)
        """
        policy_bbf = self.output_policy_bbf(state)
        policy_rftn = self.output_policy_rftn(state)
        value = self.output_value_func(state)
        return (policy_bbf, policy_rftn), value

    @torch.jit.export
    def dynamics(self, state, action, option, state_0, padding_mask: Optional[torch.Tensor] = None, offset=0):
        """
        从当前 state 和动作预测下一 state 和奖励

        Args:
            state: 前 max_len 个时刻的 state 向量 (B, L, n_hidden)
            action: 前 max_len 个时刻的动作 2 * (B, L)
            option: 前 max_len 个时刻的 option (B, L)
            state_0: 开始时刻根据真实观测的 state 向量 (B, L, n_hidden)，作为 decoder 的参数 v 进行 CA
            padding_mask:
            offset:

        Returns:
            next_state:
            reward: 奖励 (B, 1)
        """
        feature = self.ext_feat(state, action, option)
        # state_0 作为 decoder 的参数 v 用于交叉注意力
        next_state = self.dynamics_func(feature, state_0, padding_mask=padding_mask, offset=offset, dynamics=False)
        reward = self.output_reward_func(torch.reshape(feature, (feature.shape[0], -1)))  # 拉直 L 维度
        return next_state, reward

    # 初始推理（initial inference）通常指在模型开始处理一个新的输入序列或任务时的推理过程
    @torch.jit.export
    def initial_inference(self, obs, action, option, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        state_0 = self.representation(obs, action, option, padding_mask, dynamics)
        policy, value = self.prediction(torch.reshape(state_0, (state_0.shape[0], -1)))
        return policy, state_0

    # 循环推理（recurrent inference）涉及到在模型已经处理了一些输入序列或任务之后，根据当前的内部状态和之前的历史信息进行进一步的推理
    @torch.jit.export
    def recurrent_inference(self, state, action, option, state0,padding_mask: Optional[torch.Tensor] = None,offset=0):
        """
        state: (..., dim)
        action: (..., )
        option: (..., )
        """
        next_state, reward = self.dynamics(state, action, option, padding_mask=padding_mask,state_0=state0, offset=offset)
        policy, value = self.prediction(torch.reshape(next_state, (next_state.shape[0], -1)))
        return policy, next_state

    def forward(self,
                obs,
                action_prev: tuple[torch.Tensor, torch.Tensor],
                option_prev,
                padding_mask: Optional[torch.Tensor] = None,
                # n_step=None,
                gamma=0.9,
                ):
        """
        类似AC架构，第一阶段训练环境模型（dynamic函数、reward函数）和critic（value函数）
                  第二阶段训练actor（policy函数）

        Returns:
            outputs: {
                'policy': (policy_bbf, policy_rftn),
                'value': value,
                'reward': reward,
                'bis': bis,
                'rp':rp,
                'policy_train': (policy_train_bbf, policy_train_rftn),
                'policy_action_logprob': (policy_action_logprob_bbf, policy_action_logprob_rftn),
                'policy_reward': policy_reward,
                'policy_return': policy_return,
            }

        obs: (B,L,E) float
        action: (B,L) long
        option: (B,L) long
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        Outputs:
        ---------------------
        state: (B,H,L,D)
        policy: (B,H,L,D)
        value: (B,H,L,D)
        reward: (B,H,L,D)
        aux: (B,H,L,D)

        H: lookahead
        """
        # 预测窗口（目标/评估段）
        action_t = (action_prev[0][:,max_lenth:max_lenth+pre_len].clone(),
                    action_prev[1][:,max_lenth:max_lenth+pre_len].clone())
        option_t = option_prev[:, max_lenth:max_lenth + pre_len].clone()
        # 过去窗口（历史/上下文段）
        action_prev = (action_prev[0][:,0:max_lenth].clone(),
                       action_prev[1][:,0:max_lenth].clone())
        option_prev = option_prev[:, 0:max_lenth].clone()
        # 获取初始 state_0
        state_0 = self.representation(obs, action_prev, option_prev, padding_mask, dynamics=False)  # (B, max_lenth, n_hidden)


        # 教师强制
        state_t = state_0

        state_list = []
        bis_list=[]
        rp_list=[]
        value_list = []
        reward_list = []


        # Stage 1: 训练环境模型（dynamic函数、reward函数）和critic（value函数）
        for i in range(pre_len):  # 总共预测 pre_len 步
            # 获取第 i 个预测步（当前预测步）的实际动作
            action_now_bbf = torch.zeros_like(action_prev[0])
            action_now_bbf[:,0] = action_t[0][:,i]  # (B, 1)
            action_now_rftn = torch.zeros_like(action_prev[1])
            action_now_rftn[:,0] = action_t[1][:,i]
            action_now = (action_now_bbf, action_now_rftn)
            
            option_now = torch.zeros_like(option_prev)
            option_now[:,0] = option_t[:,i]  # (B, 1)

            # agent预测此步的 policy 和 value
            _, value_t = self.prediction(torch.reshape(state_t, (state_t.shape[0], -1)))
            # dynamic 基于真实动作输出下一个状态和奖励
            state_t_next, reward_t = self.dynamics(state_t, action_now, option_now, padding_mask=padding_mask, state_0=state_0, offset=1+i)#解码输出下一个状态

            # 生命体征的预测
            bis_t = self.output_bis_func(torch.reshape(state_t_next, (state_t_next.shape[0], -1)))#状态的loss需要计算
            rp_t = self.output_rp_func(torch.reshape(state_t_next, (state_t_next.shape[0], -1)))#状态的loss需要计算

            # state_list += [state_t]  # state loss计算没啥意义...，后面改了直接删了

            # Stage 1要训练的东西
            bis_list += [bis_t]
            rp_list += [rp_t]
            reward_list += [reward_t]  # 预测的奖励列表，前三条属于环境模型
            value_list += [value_t]  # 预测的价值列表，critic

            state_t = state_t_next


        # Stage 2: 训练actor（policy函数）- BC + MCTS混合方案
        # BC部分：模仿专家动作（安全、稳定）
        # MCTS部分：学习MCTS搜索找到的高价值动作（优化策略）
        state_t = state_0
        option_t = option_prev

        policy_train_list_bbf = []
        policy_train_list_rftn = []
        policy_list_bbf = []
        policy_list_rftn = []
        policy_action_logprob_list_bbf = []
        policy_action_logprob_list_rftn = []
        policy_reward_list = []
        
        # MCTS新增：用于计算MCTS guided loss的信息
        mcts_action_target_list_bbf = []  # MCTS找到的目标动作
        mcts_action_target_list_rftn = []
        mcts_weight_list = []  # 每步MCTS loss的权重（MCTS置信度）

        for i in range(pre_len):
            policy_t, value_t = self.prediction(torch.reshape(state_t, (state_t.shape[0], -1)))

            # 构造动作分布
            dist_bbf = torch.distributions.Categorical(logits=policy_t[0])  # (B, n_action)
            dist_rftn = torch.distributions.Categorical(logits=policy_t[1])
            dist = (dist_bbf, dist_rftn)
            
            # 获取真实专家动作
            action_expert_bbf = action_t[0][:, i]  # (B,)
            action_expert_rftn = action_t[1][:, i]
            
            # MCTS引导训练（批量版本）
            if self.use_mcts and self.mcts is not None and self.training:
                # 使用批量MCTS搜索
                with torch.no_grad():
                    # 批量搜索：对batch中所有样本执行MCTS
                    # 可选：使用batch_search_fast(max_samples=32)只对部分样本搜索
                    if use_mcts_batch_search:
                        # 完整批量搜索
                        actions_mcts_bbf, actions_mcts_rftn = self.mcts.batch_search(
                            state_0=state_0,
                            state_t=state_t,
                            action_prev=action_prev,
                            option_t=option_t,
                            padding_mask=padding_mask,
                            offset=1+i
                        )
                    else:
                        # 快速模式：只对部分样本搜索
                        actions_mcts_bbf, actions_mcts_rftn = self.mcts.batch_search_fast(
                            state_0=state_0,
                            state_t=state_t,
                            action_prev=action_prev,
                            option_t=option_t,
                            padding_mask=padding_mask,
                            offset=1+i,
                            max_samples=mcts_batch_samples
                        )
                    
                    # 评估MCTS动作的Q值（批量）
                    batch_size = state_t.shape[0]
                    mcts_action_bbf = torch.zeros(batch_size, dtype=torch.long, device=state_t.device)
                    mcts_action_rftn = torch.zeros(batch_size, dtype=torch.long, device=state_t.device)
                    mcts_weight = torch.zeros(batch_size, device=state_t.device)
                    
                    option_now = torch.zeros_like(option_prev)
                    option_now[:, 0] = option_t[:, i]
                    
                    # 对每个样本评估MCTS vs 专家
                    for b in range(batch_size):
                        # 评估MCTS动作
                        action_mcts_tensor = self._construct_action_tensor(
                            (actions_mcts_bbf[b].item(), actions_mcts_rftn[b].item()), 
                            (action_prev[0][b:b+1], action_prev[1][b:b+1])
                        )
                        _, reward_mcts = self.dynamics(
                            state_t[b:b+1], action_mcts_tensor, option_now[b:b+1],
                            padding_mask=padding_mask[b:b+1] if padding_mask is not None else None,
                            state_0=state_0[b:b+1], offset=1+i
                        )
                        Q_mcts = reward_mcts + gamma * value_t[b:b+1]
                        
                        # 评估专家动作
                        action_expert_tensor = self._construct_action_tensor(
                            (action_expert_bbf[b].item(), action_expert_rftn[b].item()),
                            (action_prev[0][b:b+1], action_prev[1][b:b+1])
                        )
                        _, reward_expert = self.dynamics(
                            state_t[b:b+1], action_expert_tensor, option_now[b:b+1],
                            padding_mask=padding_mask[b:b+1] if padding_mask is not None else None,
                            state_0=state_0[b:b+1], offset=1+i
                        )
                        Q_expert = reward_expert + gamma * value_t[b:b+1]
                        
                        # 如果MCTS找到更好的动作
                        if Q_mcts.item() > Q_expert.item() + use_mcts_margin:
                            mcts_action_bbf[b] = actions_mcts_bbf[b]
                            mcts_action_rftn[b] = actions_mcts_rftn[b]
                            mcts_weight[b] = 1.0
                        else:
                            mcts_action_bbf[b] = action_expert_bbf[b]
                            mcts_action_rftn[b] = action_expert_rftn[b]
                            mcts_weight[b] = 0.0
                
                # Rollout动作选择（使用第一个样本的动作）
                if teacher_forcing:
                    # 用专家动作rollout（更稳定，推荐）
                    action_for_rollout = (action_expert_bbf[0].item(), action_expert_rftn[0].item())
                else:
                    # 用MCTS动作rollout（更激进）
                    action_for_rollout = (mcts_action_bbf[0].item(), mcts_action_rftn[0].item())
            else:
                # 不使用MCTS：标准训练
                if sample_train:
                    action_t_pred = (dist[0].sample(), dist[1].sample())
                else:
                    action_t_pred = (policy_t[0].argmax(dim=-1), policy_t[1].argmax(dim=-1))
                
                action_for_rollout = (action_t_pred[0].item(), action_t_pred[1].item())
                mcts_action_bbf = action_expert_bbf  # fallback to BC
                mcts_action_rftn = action_expert_rftn
                mcts_weight = torch.zeros(action_expert_bbf.shape[0], device=state_t.device)
            
            # 构造rollout动作张量
            action_now = self._construct_action_tensor(action_for_rollout, action_prev)
            option_now = torch.zeros_like(option_prev)
            option_now[:, 0] = option_t[:, i]
            
            # Dynamics推进
            state_t_next_pred, reward_t_pred = self.dynamics(
                state_t, action_now, option_now,
                padding_mask=padding_mask, state_0=state_0, offset=1+i
            )
            
            # 计算log prob
            action_t_logprob = (
                dist[0].log_prob(torch.tensor([action_for_rollout[0]], device=state_t.device)),
                dist[1].log_prob(torch.tensor([action_for_rollout[1]], device=state_t.device))
            )

            # 收集结果
            policy_train_list_bbf += [policy_t[0]]
            policy_train_list_rftn += [policy_t[1]]
            policy_list_bbf += [policy_t[0]]
            policy_list_rftn += [policy_t[1]]
            
            policy_action_logprob_list_bbf += [action_t_logprob[0]]
            policy_action_logprob_list_rftn += [action_t_logprob[1]]
            
            policy_reward_list += [reward_t_pred.detach()] 
            
            # MCTS新增
            mcts_action_target_list_bbf += [mcts_action_bbf]
            mcts_action_target_list_rftn += [mcts_action_rftn]
            mcts_weight_list += [mcts_weight]
            
            state_t = state_t_next_pred

        # 计算贴现回报
        policy_return_list = []
        running_add = torch.zeros_like(policy_reward_list[0])
        for i in reversed(range(pre_len)):
            running_add = running_add * gamma + policy_reward_list[i]
            policy_return_list += [running_add]
        policy_return_list = policy_return_list[::-1]  # 倒转return list

        # noinspection DuplicatedCode
        bis = torch.stack(bis_list, dim=1)
        rp = torch.stack(rp_list, dim=1)
        
        policy_bbf = torch.stack(policy_list_bbf, dim=1)
        policy_rftn = torch.stack(policy_list_rftn, dim=1)

        reward = torch.stack(reward_list, dim=1)
        value = torch.stack(value_list, dim=1)

        policy_train_bbf = torch.stack(policy_train_list_bbf, dim=1)
        policy_train_rftn = torch.stack(policy_train_list_rftn, dim=1)
        
        policy_action_logprob_bbf = torch.stack(policy_action_logprob_list_bbf, dim=1)
        policy_action_logprob_rftn = torch.stack(policy_action_logprob_list_rftn, dim=1)
        
        policy_reward = torch.stack(policy_reward_list, dim=1)
        policy_return = torch.stack(policy_return_list, dim=1)
        
        # MCTS新增输出
        mcts_action_target_bbf = torch.stack(mcts_action_target_list_bbf, dim=1)  # (B, pre_len)
        mcts_action_target_rftn = torch.stack(mcts_action_target_list_rftn, dim=1)
        mcts_weights = torch.stack(mcts_weight_list, dim=1)  # (B, pre_len)

        outputs = {
            # 'state': state,  # 没有用到
            'policy': (policy_bbf, policy_rftn),
            'value': value,
            'reward': reward,
            'bis': bis,
            'rp':rp,
            'policy_train': (policy_train_bbf, policy_train_rftn),
            'policy_action_logprob': (policy_action_logprob_bbf, policy_action_logprob_rftn),
            'policy_reward': policy_reward,
            'policy_return': policy_return,
            # MCTS新增
            'mcts_action_target': (mcts_action_target_bbf, mcts_action_target_rftn),
            'mcts_weights': mcts_weights,
        }

        return outputs
    
    def _construct_action_tensor(self, action, action_prev):
        """
        辅助函数：构造动作张量用于dynamics
        
        Args:
            action: (action_bbf, action_rftn)，每个是(B,)
            action_prev: 之前的动作模板 (B, L)
            
        Returns:
            (action_bbf_tensor, action_rftn_tensor)，每个是(B, L)，只有第0位是当前动作
        """
        action_bbf, action_rftn = action
        action_now_bbf = torch.zeros_like(action_prev[0])
        action_now_rftn = torch.zeros_like(action_prev[1])
        action_now_bbf[:, 0] = action_bbf
        action_now_rftn[:, 0] = action_rftn
        return (action_now_bbf, action_now_rftn)

    # # not used
    # @torch.jit.export
    # def auxiliary(self, state):
    #     aux = self.output_aux_func(state)
    #     return aux
