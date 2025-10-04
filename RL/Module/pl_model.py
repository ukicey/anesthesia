#!/usr/bin/env python3

import torchmetrics
from torch import nn

from pathlib import Path
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from RL.utils import support_to_scalar, logit_regression_loss, logit_regression_mae, masked_loss, masked_mean,cross_entropy,cross_entropy_mae
from constant import *


class BaseModule(pl.LightningModule):
    def __init__(self,  model, output_dir=None,
                 lr=lr, model_args=None, data_args=None, module_args=None, data_module=None,
                 ):
        super().__init__()

        self.lr = lr
        self.model_name = "test"
        self.model_args = {}
        self.model = model  # 所有nn.Module的子类都会被优化器识别，因此可以使用model.parameters()来获取所有参数
        self.data_args = {}
        self.module_args ={}
        self.data_module = {}


        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # metic
        self.train_loss = None
        metrics = {
            'loss_bis': torchmetrics.MeanMetric(),
            'loss_rp': torchmetrics.MeanMetric(),
            'loss_action': torchmetrics.MeanMetric(),
            'loss_bc': torchmetrics.MeanMetric(),  # BC损失
            'loss_mcts': torchmetrics.MeanMetric(),  # MCTS损失

            'loss_value': torchmetrics.MeanMetric(),
            'loss_reward': torchmetrics.MeanMetric(),

            'action_mae': torchmetrics.MeanMetric(),
            'value_mae': torchmetrics.MeanMetric(),
            'rp_mae': torchmetrics.MeanMetric(),
            'bis_mae': torchmetrics.MeanMetric(),
            'reward_mae': torchmetrics.MeanMetric(),
        }
        self.metrics = nn.ModuleDict(metrics)  # 通过使用字典的键访问相应的模块，你可以计算相应的损失和准确度值。

        self.n_step = self.model_args.get('n_step', max_lenth)  # 获取n_step参数，如果没有则默认为1

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # Adam 算法
        return optimizer

    def on_train_start(self):
        log_hyperparams = {
            "lr": self.lr,
        }
        log_hyperparams.update(self.model_args)
        self.logger.log_hyperparams(log_hyperparams)

    # def on_train_end(self) -> None:
    #     torch.cuda.empty_cache()

    # def on_validation_end(self) -> None:
    #     torch.cuda.empty_cache()

    @staticmethod
    def label_data_stack(label_data, n_step):
        keys = [
            'action',
            'mask_reward',
            'bis_target',
            'rp_target',
            'reward',
            'cumreward',
        ]

        # create mask
        mask = torch.ones_like(label_data['bis_target'])
        masks = []
        for i in range(n_step):
            mask_t = torch.roll(mask, shifts=-i, dims=1)
            mask_t[:, -1] = 0
            masks += [mask_t]
            mask = mask_t
        stack_mask = torch.stack(masks, dim=1)

        # create stack label
        stack_label = {}
        for key in keys:
            val = label_data[key]
            vals = []
            bbf_list = []
            rftn_list = []
            for i in range(n_step):
                if key == 'action':
                    bbf_val = val[:, 0, max_lenth + i]
                    rftn_val = val[:, 1, max_lenth + i]
                    bbf_list += [bbf_val]
                    rftn_list += [rftn_val]
                    pass
                else :
                    val_t = val[:,max_lenth+i,...]#大小为batch,max_length,xxx
                    vals += [val_t]
            if key == 'action':
                bbf_val = torch.stack(bbf_list, dim=1)
                rftn_val = torch.stack(rftn_list, dim=1)
                stack_label[key] = (bbf_val, rftn_val)
                pass
            else: 
                val = torch.stack(vals, dim=1)
                stack_label[key] = val

        stack_label['padding'] = label_data['padding']

        return stack_label, stack_mask


class RLSLModelModule(BaseModule):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.loss_state_weight = self.module_args.get('loss_state_weight', 0.1)
        self.loss_rl_weight = self.module_args.get('loss_rl_weight', 1.0)
        self.loss_joint = self.module_args.get('loss_rl_joint', True)
        
        # 训练策略配置
        self.warmup_epochs = warmup_epochs
        self.env_loss_weight = env_loss_weight

    def forward(self, x):
        """

        Args:
            x:

        Returns:
            out (dict): {
                'policy': (action_bbf, action_rftn),  # (b, n_step, l, action_dim)
                'policy_train': (action_bbf_train, action_rftn_train),  # (b, n_step, l, action_dim)
                'state': state,  # (b, n_step, l, state_dim)
                'value': value,  # (b, n_step, l, 1)
                'reward': reward,  # (b, n_step, l, 1 or reward_support_size)
                'bis': bis,  # (b, n_step, l, 1 or bis_support_size)
                'rp': rp,  # (b, n_step, l, 1 or rp_support_size)
            }
        """
        out = self.model(*x)
        return out

    def get_loss(self, pred_data, label_data):
        action = label_data['action']
        mask_reward = label_data['mask_reward']  # (b, l)
        rp_target = label_data['rp_target']
        bis_target = label_data['bis_target']
        reward = label_data['reward']
        cumreward = label_data['cumreward']
        padding = label_data['padding']

        pred_policy = pred_data['policy']
        pred_policy_train = pred_data['policy_train']
        pred_bis = pred_data['bis']
        pred_rp = pred_data['rp']
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']

        b, n_step, length = pred_policy[0].shape[:3]#截取前三个维度的
        device = pred_policy[0].device

        # mask
        step_mask = torch.ones([length], device=device)  # (batch, )
        step_masks = [step_mask]
        for i in range(n_step - 1):
            step_mask = torch.cat([torch.zeros_like(step_mask[-1:]), step_mask[:-1]])
            step_masks += [step_mask]
        step_mask = torch.stack(step_masks)  # [n_step, len]
        step_mask = step_mask[None, ...].expand(b, n_step, length)  # (b, n, l) 斜方一半掩盖
        step_mask = step_mask.bool()
        # action function - BC loss（行为克隆，模仿专家）
        pred_action_bbf = pred_policy[0]
        pred_action_train_bbf = pred_policy_train[0]
        loss_bc_bbf, true_action_bbf = cross_entropy(pred_action_train_bbf, action[0])
        
        pred_action_rftn = pred_policy[1]
        pred_action_train_rftn = pred_policy_train[1]
        loss_bc_rftn, true_action_rftn = cross_entropy(pred_action_train_rftn, action[1])
        
        loss_bc = loss_bc_bbf + loss_bc_rftn  # BC总损失
        
        # MCTS loss（学习MCTS搜索找到的高价值动作）
        if use_mcts and 'mcts_action_target' in pred_data:
            mcts_action_target = pred_data['mcts_action_target']  # (bbf, rftn)
            mcts_weights = pred_data['mcts_weights']  # (B, pre_len)
            
            # 计算MCTS动作的交叉熵
            loss_mcts_bbf, _ = cross_entropy(pred_action_train_bbf, mcts_action_target[0])
            loss_mcts_rftn, _ = cross_entropy(pred_action_train_rftn, mcts_action_target[1])
            
            # 加权MCTS loss（只在MCTS找到更好动作时计入）
            # mcts_weights: (B, pre_len)，需要扩展到匹配loss shape
            mcts_weights_expanded = mcts_weights.unsqueeze(-1)  # (B, pre_len, 1)
            loss_mcts = (loss_mcts_bbf * mcts_weights_expanded).mean() + (loss_mcts_rftn * mcts_weights_expanded).mean()
            
            # 混合损失：BC + MCTS
            loss_action = lambda_bc * loss_bc + lambda_mcts * loss_mcts
        else:
            loss_mcts = torch.tensor(0.0, device=device)
            loss_action = loss_bc
        
        if not self.loss_joint:
            loss_reward = 0
            loss_state = 0
            loss = loss_action
        else:
            # value function
            if pred_value.size(-1) > 1:#单一value输出
                loss_value = logit_regression_loss(pred_value, cumreward, mask=step_mask)#回报和vaule,其实就是return
            else:
                loss_value = masked_loss(F.mse_loss, pred_value.squeeze(-1), cumreward, mask=None)

            # reward function
            mask_reward = mask_reward.expand(b, n_step).bool()
            if pred_reward.size(-1) > 1:
                loss_reward = logit_regression_loss(pred_reward, reward, mask=step_mask * mask_reward)
            else:
                loss_reward = masked_loss(F.mse_loss, pred_reward.squeeze(-1), reward,
                                          mask=None)

            

            # live loss
            # bis和rp loss

            loss_bis,true_bis=cross_entropy(pred_bis,bis_target)
            #loss_bis = masked_loss(F.mse_loss, pred_bis.squeeze(-1), bis_target, mask=None)

            loss_rp,true_rp = cross_entropy(pred_rp, rp_target)
            #loss_rp = masked_loss(F.mse_loss, pred_rp.squeeze(-1), rp_target, mask=None)

            # 训练策略：Warm-up + 加权
            loss_env = loss_bis + loss_rp + loss_value + loss_reward
            
            if self.current_epoch < self.warmup_epochs:
                # Phase 1: Warm-up阶段，只训练环境模型
                loss = loss_env
                # 记录：warmup阶段不训练policy
                in_warmup = True
            else:
                # Phase 2: 联合训练，但环境模型loss权重更大
                loss = loss_action + self.env_loss_weight * loss_env
                in_warmup = False
            
            # 计算并打印各个损失及其加权总和
            # print("Action:", str(loss_action) +
            #   " Rp:", str(loss_rp) +
            #   " Bis:", str(loss_bis) +
            #   " Value:", str(loss_value) +
            #   " Reward:", str(loss_reward) +
            #   " Warmup:", str(in_warmup))

        losses = {
            '_step_mask': step_mask,
            '_mask_reward': mask_reward,
             '_pred_bis': pred_bis,
            '_true_bis': true_bis,
             '_pred_rp': pred_rp,
            '_true_rp': true_rp,
            '_pred_action': (pred_action_bbf,pred_action_rftn),
            '_true_action': (true_action_bbf,true_action_rftn),

            'loss': loss,
            'loss_action': loss_action,
            'loss_bc': loss_bc,  # BC损失
            'loss_mcts': loss_mcts,  # MCTS损失
            'loss_value': loss_value,
            'loss_reward': loss_reward,
            'loss_rp': loss_rp,
            'loss_bis':loss_bis,
        }

        return losses

    #  类似于pytorch中的forward函数，解决的是预测与损失计算，并返回预测与损失结果。

    def training_step(self, batch, batch_idx):#batch为fit管理传入的数据
        x, y = batch  # return (obs, action_prev, option, padding), label_data
        label_data = y
        pred_data = self.model(*x)#向前传播,得到预测结果 传入三维 第一维是batch
        b, n_step, length = pred_data['policy'][0].shape[:3]
        label_data, stack_mask = self.label_data_stack(label_data, n_step)
        losses = self.get_loss(pred_data, label_data)#预测值和标签值计算损失
        loss = losses['loss']
        train_loss=losses['loss_bis']
        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, label_data = batch
        pred_data = self.model(*x)
        b, n_step, length = pred_data['policy'][0].shape[:3]#取前三个维度
        label_data, stack_mask = self.label_data_stack(label_data, n_step)
        losses = self.get_loss(pred_data, label_data)
        loss = losses['loss']

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        
        # 记录是否在warmup阶段
        in_warmup = self.current_epoch < self.warmup_epochs
        self.log("in_warmup", float(in_warmup), prog_bar=True, sync_dist=True)

        action = label_data['action']

        bis_target = label_data['bis_target']
        rp_target=label_data['rp_target']
        reward = label_data['reward']
        mask_reward = label_data['mask_reward']  # (b, l)
        cumreward = label_data['cumreward']

        pred_policy = pred_data['policy']
        pred_bis = pred_data['bis']
        pred_rp = pred_data['rp']
        # pred_state = pred_data['state']  # 完全没有用到
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']
        b, n_step, length = pred_policy[0].shape[:3]
        device = pred_policy[0].device

        # mask
        step_mask = losses['_step_mask']
   
        mask_reward = losses['_mask_reward']

        # action function
        pred_action = losses['_pred_action']
        true_action = losses['_true_action']
        

        # metrics action
        action_mae_bbf=cross_entropy_mae(pred_action[0], true_action[0])
        action_mae_rftn=cross_entropy_mae(pred_action[1], true_action[1])

        # metrics value mae
        if pred_value.size(-1) > 1:
            value_mae = logit_regression_mae(pred_value, cumreward, mask=step_mask)
        else:
            value_mae = masked_loss(F.l1_loss, pred_value.squeeze(-1), cumreward, mask=step_mask)

        # metrics reward mae
        if pred_reward.size(-1) > 1:
            reward_mae = logit_regression_mae(pred_reward, reward, mask=step_mask * mask_reward)
        else:
            reward_mae = masked_loss(F.l1_loss, pred_reward.squeeze(-1), reward, mask=None)

        #def cross_entropy_mae(pred,true_action):
        pred_bis = losses['_pred_bis']
        true_bis = losses['_true_bis']
        pred_rp = losses['_pred_rp']
        true_rp = losses['_true_rp']
        bis_mae=cross_entropy_mae(pred_bis,true_bis)
        rp_mae=cross_entropy_mae(pred_rp,true_rp)

        #bis_mae = masked_loss(F.l1_loss, pred_bis.squeeze(-1), bis_target,mask=None)
        #rp_mae = masked_loss(F.l1_loss, pred_rp.squeeze(-1), rp_target, mask=None)

        name_to_value = {
            'loss_action': losses['loss_action'],
            'loss_bc': losses['loss_bc'],  # BC损失
            'loss_mcts': losses['loss_mcts'],  # MCTS损失
            'loss_value': losses['loss_value'],
            'loss_reward': losses['loss_reward'],
            'loss_bis': losses['loss_bis'],
            'loss_rp': losses['loss_rp'],
            'action_mae': (action_mae_bbf, action_mae_rftn),
            'value_mae': value_mae,
            'reward_mae': reward_mae,
            'bis_mae': bis_mae,
            'rp_mae': rp_mae,
        }

        for metrics_name, value in name_to_value.items():
            metrics = self.metrics[metrics_name]
            metrics.update(value)
            metrics_attr_name = f"val_{metrics_name}"
            self.log(metrics_attr_name, metrics, prog_bar=False, sync_dist=True, metric_attribute=metrics_attr_name)

        return loss