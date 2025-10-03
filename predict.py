import json
import os
from pathlib import Path


import pandas as pd

from RL.Module.data import TimeseriesExtractor, ActionExtractor

if (__name__ == '__main__') or (__package__ == ''):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
from RL.pip import *

from RL.agent import ArmAgent
# else:
#     from .datasets.pipe import DiabetesPipeline
#     from .datasets.ts_dataset import TSRLDataset
#     from .models.agent import InsulinArmAgent


from torch.utils.data.dataloader import default_collate
from constant import *

def get_sample(df,df_meta_path,_map,gamma):

    feat_extractor = TimeseriesExtractor(df_meta_path,
                                         add_mask=add_mask)
    # 打药执行序列
    action_extractor = ActionExtractor()
    
    #把df下标从max_length到

  
     
    #建立索引
    df_case_raw = df.reset_index(drop=True)
    # 建立新列,名字为time,为索引1,2,3..
    df_case_raw['timestamps'] = np.arange(len(df_case_raw))
    gamma = float(gamma)
    # extract features
    df_case_feat = feat_extractor.transform(df_case_raw)
    df_obs = df_case_feat
    # 把label_*列的值找出来,汇集成列表
    obs = df_obs.fillna(0).values
    l = len(obs)
    df_aux = pd.DataFrame({'Column': [1] * l})
# 找出缺失表,把缺失的值填充为0,不缺失的值填充为1
    df_mask_aux = df_aux.notna().astype('int')
    # get the value,对nan值进行填充
    # get the value,对nan值进行填充
    obs = df_obs.fillna(0).values

    aux = df_aux.fillna(0).values
    mask_aux = df_mask_aux.values
    result = {
        'obs': obs,
        'aux': aux,
        'mask_aux': mask_aux,
    }

    df_case_raw['reward'] = cal_reward(df_case_feat[cols_state], dtype='risk',f_map=_map)
    df_case_raw['return'] = reward2return(df_case_raw['reward'], gamma=gamma)
    # 计算每一步损失
    df_case_raw = action_extractor.transform(df_case_raw)  # add action and option

    df_reward = df_case_raw['reward']
    df_return = df_case_raw['return']
    df_action = (df_case_raw['action_BBF'], df_case_raw['action_RFTN'])
    df_option = df_case_raw['option']
    # 填充值
    reward = df_reward.fillna(0).values
    cumreward = df_return.fillna(0).values
    action_BBF = df_action[0].fillna(0).values
    action_RFTN = df_action[1].fillna(0).values
    option = df_option.fillna(0).values

    df_mask_action = (df_action[0] > 0).astype('int')
    mask_action = df_mask_action.values

    bis = df_case_feat['BIS'].values#glu_target格式?
    rp= df_case_feat['MAP|MAP'].values#glu_target格式?

    result.update({
            'reward': reward,
            'cumreward': cumreward,
            'action': (action_BBF, action_RFTN),
            'option': option,
            'mask_action': mask_action,
            'BIS': bis,#生命体征
            'MAP|MAP':rp
    })

        # padding: [:l] is False, [l:] is True
    l = len(obs)
    padding = np.zeros(l, dtype='bool')
   #对于


    return result

def dataframe2tensor(sample):


    #获取tensor
    # 在前面部分获取sample(包括obs,aux,mask_aux,padding,live,reward,cumreward,action,option,mask_action)
    obs = sample['obs'].astype('float32')
    aux = sample['aux'].astype('float32')  # for BCE loss
    aux = np.roll(aux, -1, axis=0)
    aux[-1] = 0
    mask_aux = sample['mask_aux'].astype('int')
    mask_aux = np.roll(mask_aux, -1, axis=0)
    mask_aux[-1] = 0
    bis = sample['BIS'].astype('int')
    BIS_target = bis # move forward one point of time as a label

    rp = sample['MAP|MAP'].astype('int')
    rp_target = rp  # move forward one point of time as a label


    reward = sample['reward'].astype('float32')


    cumreward = sample['cumreward'].astype('float32')


    action_BBF = sample['action'][0].astype('int')
    action_RFTN = sample['action'][1].astype('int')
    action_BBF = action_BBF.clip(0, n_action)#对action切片
    action_RFTN = action_RFTN.clip(0, n_action)#对action切片
    
    BIS_target= BIS_target.clip(0, n_bis)
    rp_target=rp_target.clip(0, n_rp)
    
    action_prev = (np.roll(action_BBF, shift=1), np.roll(action_RFTN, shift=1))  # 空出第一个位置
    action_prev[0][0] = 0
    action_prev[1][0] = 0
    action=action_prev
    
    option = sample['option'].astype('int')
    option = option.clip(0, 4)
    mask_reward = bis > 0.01
    mask_reward[-1] = False



    return (obs, action_prev, option, BIS_target,rp_target)  # 二维



class Model(object):
    def __init__(self, model_dir, df_meta_path, beam_size=2, device=None):
        super(Model, self).__init__()
        self.df_meta_path = df_meta_path
        self.beam_size = beam_size
        self.model_dir = model_dir

        self.agent = ArmAgent.build_from_dir(model_dir, device)
        self.size = self.agent.max_len
    
    def predict_v1(self,df_meta_path, df, beam_size=None,name=None):
        first_map=df["MAP|MAP"][0]
        end_index=start_len
        import torch

        df_pred = pd.DataFrame()
        df_pred_ac_t_bbf=torch.tensor([0, 0, 0, 0,0, 0, 0, 0, 0, 0])
        df_pred_ac_t_rftn=torch.tensor([0, 0, 0, 0,0, 0, 0, 0, 0, 0])
        df_pred_bis_t=torch.tensor([0, 0, 0, 0,0, 0, 0, 0, 0, 0])
        df_pred_pr_t=torch.tensor([0, 0, 0, 0,0, 0, 0, 0, 0, 0])
        df_target = pd.DataFrame()
        df_target['action_BBF']=df['BBF_SPEED']
        df_target['action_RFTN']=df['RFTN_SPEED']
        df_target['bis']=df['BIS']
        df_target['map']=df['MAP|MAP']
        df_target['action_BBF'] = np.roll( df_target['action_BBF'], shift=1)  # 空出第一个位置
        df_target['action_BBF'][0] = 0
        df_target['action_RFTN'] = np.roll( df_target['action_RFTN'], shift=1)  # 空出第一个位置
        df_target['action_RFTN'][0] = 0
        padding = np.zeros(len(df), dtype='bool')
        
        
        for i in range(end_index, len(df), pre_len):
            if(i<max_lenth):
                true_data=df.iloc[0:i].copy()
                true_data = pd.concat([true_data,pd.DataFrame(index=range(max_lenth-i))], axis=0)#随机填充,增强对开始任务的学习能力量


                selected_padding= np.copy(padding[0:i])
            
                selected_padding =  np.concatenate([ selected_padding,np.ones(max_lenth-i, dtype='bool')])

            else:
                true_data=df.iloc[i-max_lenth:i].copy()

                selected_padding= np.copy(padding[i-max_lenth:i])
            

   
        
            sample = get_sample(true_data,df_meta_path,first_map,gamma)
            r=[]
            # 例如，打印每个窗口的内容
            tensors = dataframe2tensor(sample)
            tensors = default_collate([tensors])#堆叠张量
            obs, action, option, BIS_target,rp_target = tensors


            # convert to tensor
            # predict
            beam_size = beam_size if beam_size is not None else self.beam_size
            # 记录开始时间
            import time
            start_time = time.time()
            output = self.agent.self_rollout(obs, action, option,selected_padding, beam_size=beam_size)  # extra one day
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"代码执行花费的时间: {execution_time}秒")
            actions_out, bis_out,rp_out = output
            
            mean_dose_BBF =torch.mean(actions_out[0].float())
            mean_dose_BBF =torch.full(size=(1,5),fill_value=mean_dose_BBF)
            mean_dose_RFTN =torch.mean(actions_out[1].float())
            mean_dose_RFTN =torch.full(size=(1,5),fill_value=mean_dose_RFTN)
            
            #padding_series = pd.Series([0] * max_lenth)
            df_pred_ac_t_bbf= torch.cat([df_pred_ac_t_bbf , mean_dose_BBF[0,:]], dim=0)
            df_pred_ac_t_rftn= torch.cat([df_pred_ac_t_rftn , mean_dose_RFTN[0,:]], dim=0)

            df_pred_bis_t=torch.cat([df_pred_bis_t ,  bis_out[0,:]], dim=0)
            df_pred_pr_t=torch.cat([df_pred_pr_t ,  rp_out[0,:]], dim=0)
        df_pred['action_BBF']=df_pred_ac_t_bbf
        df_pred['action_RFTN']=df_pred_ac_t_rftn
        df_pred['bis']=df_pred_bis_t
        df_pred['map']=df_pred_pr_t
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 2 * 3))
       
        for i, col in enumerate(df_target.columns):
            # 获取当前子图
            ax = axes[i]

            # 绘制折线图,体现两个版本的差异
            ax.plot(df_target[col].to_numpy(), label="true")
            ax.plot(df_pred[col].to_numpy(), label="pred")

            # 设置子图标题
            ax.set_title(col)

            # 添加标签和标题等
            ax.set_xlabel('min')
            ax.set_ylabel('level')

            # 显示图例
            ax.legend()

        # 调整子图的布局
        plt.tight_layout()
        # 保存图形
        plt.savefig(raw_p+'output/picture/'+name+'.png')
        return r




def predict(model_name,  beam_size=5):
    #mode1:val=0
    if model_val==0:
        csv_files = [file for file in os.listdir(df_path) if file.endswith('.csv')]
        # 假设 csv_files 是你要处理的 CSV 文件名列表

        csv_files_length = len(csv_files)
        percent_to_keep = rate_csv
        num_to_keep = int(csv_files_length * percent_to_keep)
        # 只保留1%的数据
        csv_files = csv_files[:num_to_keep]
        # 按比例分割csv_files
        n_train = int(len(csv_files) * train_r)
        train_list = csv_files[:n_train]
        valid_list = csv_files[n_train:]
        #只保留valid中10%的数据
        valid_list = valid_list[:int(len(valid_list) * val_rate)]
        model = Model(model_dir=model_dir, df_meta_path=data_kargs_default['df_meta_path'])
        #保留valid_list前五个项
        #遍历valid_list
        valid_list = valid_list[:5]
        for file in valid_list:
            df_data = pd.read_csv(os.path.join(df_path,file))
            if(df_data.shape[0] >pre_len+1):
                model.predict_v1(df=df_data,df_meta_path=data_kargs_default['df_meta_path'],beam_size=beam_size,name=file)
                print(file+"预测完成")
            
    #
    if model_val==1:
        df_data = pd.read_csv("/data/tongqi/anesthesia/DATA/Valid_Data/范庆生_V2.csv")
        model = Model(model_dir=model_dir, df_meta_path=data_kargs_default['df_meta_path'])
        model.predict_v1(df=df_data,df_meta_path=data_kargs_default['df_meta_path'],beam_size=beam_size,name='1_valid_data')


   
    
    
        print("预测完成")
    # 打开一个新的CSV文件，如果文件不存在将会被创建
  

    # for file in valid_list:
    #     df_data = pd.read_csv(os.path.join(df_path, file))
    #     model = Model(model_dir=model_dir, df_meta_path=data_kargs_default['df_meta_path'])
    #     result = model.predict(df_data,df_meta_path=data_kargs_default['df_meta_path'],beam_size=beam_size)




if __name__ == '__main__':

    import fire

    result = fire.Fire(predict)