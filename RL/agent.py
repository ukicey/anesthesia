import warnings

import numpy as np
import pandas as pd
from pathlib import Path

import json
import torch


def support_to_scalar(logits):#reward的概率
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=-1)
    support_size = logits.shape[-1] // 2
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
            .expand(probabilities.shape)
            .float()
            .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=-1)
    return x




from constant import *
class ArmAgent(object):
    def __init__(self, config_path, model_path, device=None):
        super(ArmAgent, self).__init__()
        if isinstance(model_path, str) or isinstance(model_path, Path):
            model_path = Path(model_path)
            if model_path.name == 'scripted_model.zip':
                self.model = torch.jit.load(str(model_path))
            else:
                torch.cuda.empty_cache()
                self.model = torch.load(str(model_path), map_location=torch.device('cpu'))
        else:
            self.model = model_path

        if config_path is None:
            warnings.warn('config_path is None. using default ')
            self.config = {
                'max_len': max_lenth,
                'gamma': gamma,
            }
        else:
            self.config = json.loads(Path(config_path).read_text())

        self.max_len = max_lenth
        if 'max_len' in self.config:
            self.max_len = self.config['max_len']
        elif 'max_seq_len' in self.config:
            self.max_len = self.config['max_seq_len']
        else:
            warnings.warn(f'max_len not found in config: {config_path}. using 128')
        self.gamma = self.config['gamma']

        self.device = 'cpu'

        if 'n_input' in self.config:
            self.n_input = self.config['n_input']
        else:
            self.n_input = self.model.n_input
        self.model.eval()
        if self.device is not None:
            self.model.to(device)

    @staticmethod
    def build_from_dir(model_dir, device=None):
        model_dir = Path(model_dir)
        if (model_dir / 'scripted_model.zip').exists():
            model_path = str(model_dir / 'scripted_model.zip')
            config_path = model_dir / 'scripted_model.config.json'
        else:
            model_path = str(model_dir / 'model.pt')
            config_path = None
        agent = ArmAgent(config_path, model_path, device)
        return agent

    def self_rollout(self, obs, action_all, option_all,padding, beam_size=5):
       
        '''
        obs: np.array
            torch.Size([1, 128, 381])
        action_all: np.array
            torch.Size([1, 128])
        option_all: np.array
            torch.Size([1, 128])
        t0: int
            [0,t0): observed time point
        tt: int
            [t0,tt): time point to be predicted

        NOTE:
            [0,t0): observed time point
            [t0,tt): time point to be predicted
            [tt,max_len): padding

        '''
        model = self.model

        shape_valid = self.max_len, self.n_input
        assert len(obs.shape) == 3
        #obs是预测集

        if obs.shape[1:] != shape_valid:
            print(obs.shape[1:], shape_valid)
        assert obs.shape[1:] == shape_valid

        # obs = torch.tensor(obs, dtype=torch.float32)
        # option_all = torch.tensor(option_all, dtype=torch.int)


        device = torch.device("cpu")
        model.to(device)
        
        action_pred_bbf=[]
        action_pred_rftn=[]
        bis_pred=[]
        rp_pred=[]
        padding=torch.from_numpy(padding)
        padding = padding.unsqueeze(0)

        # initialization
        with torch.no_grad():
            policy0, state0 = model.initial_inference(obs, action_all, option_all, padding)
            
        action_bbf_0 = policy0[0].argmax(dim=-1)#初始动作
        action_bbf_1=action_bbf_0
        action_rftn_0 = policy0[1].argmax(dim=-1)#初始动作
        action_rftn_1=action_rftn_0
        
        state_temp=state0

        # iteration
        with torch.no_grad():
           
            for i in range(pre_len):
                action_pred_bbf.append(action_bbf_1)
                action_pred_rftn.append(action_rftn_1)
                               
                action_bbf_now = torch.zeros_like(action_all[0])
                action_rftn_now = torch.zeros_like(action_all[1])
                action_bbf_now[:,0] = action_bbf_1[0]#只采样之前的动作?
                action_rftn_now[:,0] = action_rftn_1[0]
                action_now = (action_bbf_now,action_rftn_now)
                
                option_now = torch.zeros_like(option_all)
                option_now[:,0] = 1

                # calculate 
                # value, reward, policy, next_state = model.recurrent_inference(state_temp, action_prev, option_temp)
                policy, next_state = model.recurrent_inference(state_temp, action_now, option_now, state0,padding,offset=1+i)
                bis_t = model.output_bis_func(torch.reshape(next_state, (next_state.shape[0], -1)))#状态的loss需要计算
                rp_t = model.output_rp_func(torch.reshape(next_state, (next_state.shape[0], -1)))#状态的loss需要计算
                state_temp=next_state
                bis_pred.append(bis_t.argmax(dim=-1))
                rp_pred.append(rp_t.argmax(dim=-1))
                action_bbf_1=policy[0].argmax(dim=-1)#下一个动作
                action_rftn_1=policy[1].argmax(dim=-1)#下一个动作
                
        action_BBF_pred = torch.stack(action_pred_bbf, dim=1)
        action_RFTN_pred = torch.stack(action_pred_rftn, dim=1)
        bis = torch.stack(bis_pred, dim=1)
        rp=torch.stack(rp_pred, dim=1)

        return (action_BBF_pred, action_RFTN_pred),bis,rp




