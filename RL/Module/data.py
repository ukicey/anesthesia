"""
该文件整体负责将单病例 CSV 序列加工为模型可用的张量样本，
并通过 Lightning 的 DataModule 组织训练与验证数据加载
"""

import os
import shutil

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from RL.pip import *
from constant import *


class TimeseriesExtractor(object):
    """
    时间序列特征提取器：
    - 基于元数据 `df_meta_path` 对输入病例 DataFrame 做缺失值填充与独热编码
    - 将每个原始特征的“是否缺失”转为 0/1 掩码列，并追加到特征中（列名后缀为 `_notna`）
    - 提供 `feature_names` 与 `n_features` 以便上游模型获知输入维度
    """

    def __init__(self, df_meta_path, add_mask):
        super(TimeseriesExtractor, self).__init__()
        self.df_meta_path = df_meta_path
        self.add_mask = add_mask


    def transform(self, df_case):
        # 获得动作特征编码
        df_feat, df_mask = pip_fillna(df_case, self.df_meta_path)#不可变序列?
        df_feat = onehot(df_feat, self.df_meta_path)

        df_feat = pd.concat([df_feat, df_mask.add_suffix('_notna').astype('int')], axis=1)

        return df_feat

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    @lru_cache(1)
    def feature_names(self):
        """
        obtain category information for all timegroups
        """
        df_dummy = pd.DataFrame({})
        df_dummy = self.transform(df_dummy)
        return list(df_dummy.columns)
class ActionExtractor(object):
    """
    extract and complete the category and medication information of insulin
    """

    def __init__(self):
        super(ActionExtractor, self).__init__()
        self.col_action = col_action
        self.option = option
        self.option_map = option_map
    def transform(self, df_case):
        #获得动作特征编码
        df_case_labeled = df_case


        df_case_labeled['action_BBF'] = df_case_labeled[self.col_action[0]].fillna(0).astype('int')
        df_case_labeled['action_RFTN'] = df_case_labeled[self.col_action[1]].fillna(0).astype('int')
        df_case_labeled['MAP|MAP']=df_case_labeled['MAP|MAP'].round().fillna(0).astype('int')
        df_case_labeled['BIS']=df_case_labeled['BIS'].fillna(0).round().astype('int')

        if self.option in df_case_labeled:
            # map, supplement, and normalize options
            df_case_labeled['option'] = df_case_labeled[self.option].map(option_map).fillna(0).astype('int')
        else:
            df_case_labeled['option'] = 1
            df_case_labeled.loc[df_case_labeled['BIS'].isna(), 'option'] =0

        return df_case_labeled





class TSRLDataset(Dataset):
    #timegroup,7时间组的数量,data_dir全部数据
    def __init__(self, csv_files,df_meta_path, max_seq_len, brl_setting, reward_dtype, return_y, feat_append_mask):
        self.df_meta_path = df_meta_path
        self.max_seq_len = max_seq_len
        self.brl_setting = brl_setting
        self.reward_dtype = reward_dtype
        self.return_y = return_y
        self.feat_append_mask = feat_append_mask
        self.gamma = gamma
        self.n_reward_max = n_reward_max
        self.n_value_max = n_value_max
        self.n_action = n_action
        self.n_option = n_option
        self.df_array = []
        self.cache_dir = cache_dir
        self.cache_renew = cache_renew
        self.diskcache = diskCache
        self.cols_state = cols_state#生命体征集合
        self.cols_label = cols_aux#辅助任务

        #遍历csv_files
        for file in csv_files:
            df = pd.read_csv(os.path.join(df_path, file))
            if(df.shape[0] >pre_len+start_len):
                self.df_array.append(df)
            # process data_dir, df_meta_pa
        print(len(self.df_array))
        feat_extractor = TimeseriesExtractor(df_meta_path,
                                             add_mask=add_mask)
        # 打药执行序列
        action_extractor = ActionExtractor()
        self.feat_extractor = feat_extractor
        self.action_extractor = action_extractor

    @property
    def n_features(self):
        #返回
        return self.feat_extractor.n_features

    @property
    def n_labels(self):
        return len(self.labels)

    @property
    def labels(self):
        return self.cols_label

    def __len__(self):
        return len(self.df_array)

    def _get_sample(self, df_case_raw:pd.DataFrame, max_seq_len, gamma, reward_dtype,map):
        
        
        #建立索引
        df_case_raw = df_case_raw.reset_index(drop=True)
        # 建立新列,名字为time,为索引1,2,3..
        df_case_raw['timestamps'] = np.arange(len(df_case_raw))
        gamma = float(gamma)
        # extract features
        df_case_feat = self.feat_extractor.transform(df_case_raw)
        df_obs = df_case_feat
        # 把label_*列的值找出来,汇集成列表
        df_aux =  pd.DataFrame({'Column': [1] * max_lenth})
        # 找出缺失表,把缺失的值填充为0,不缺失的值填充为1
        df_mask_aux = df_aux.notna().astype('int')

        # get the value,对nan值进行填充
        obs = df_obs.fillna(0).values
        aux = df_aux.fillna(0).values
        #
        mask_aux = df_mask_aux.values
        result = {
            'obs': obs,
            'aux': aux,
            'mask_aux': mask_aux,
        }
        

        df_case_raw['reward'] = cal_reward(df_case_feat[self.cols_state], dtype=reward_dtype,f_map=map)
        df_case_raw['return'] = reward2return(df_case_raw['reward'], gamma=gamma)
        # 计算每一步损失
        df_case_raw = self.action_extractor.transform(df_case_raw)  # add action and option

        df_reward = df_case_raw['reward']
        df_return = df_case_raw['return']
        df_action = (df_case_raw['action_BBF'], df_case_raw['action_RFTN'])
        df_option = df_case_raw['option']
        # 填充值
        reward = df_reward.fillna(0).values
        cumreward = df_return.fillna(0).values
        action = (df_action[0].fillna(0).values, df_action[1].fillna(0).values)
        option = df_option.fillna(0).values

        df_mask_action = (df_action[0] > 0).astype('int')
        mask_action = df_mask_action.values

        bis = df_case_feat['BIS'].values#glu_target格式?
        rp= df_case_feat['MAP|MAP'].values#glu_target格式?

        result.update({
            'reward': reward,
            'cumreward': cumreward,
            'action': action,
            'option': option,
            'mask_action': mask_action,
            'BIS': bis,#生命体征
            'MAP|MAP':rp
        })

            # padding: [:l] is False, [l:] is True
        



        return result
    
    def __getitem__(self, idx):  # 数据集迭代器会返回这个数据
        import diskcache
        if not (self.cache_dir is None):
            if self.diskcache is None:
                if self.cache_renew:
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                self.diskcache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')
        df= self.df_array[idx].copy()
        #把base|gender列中为0的值换成M,1换成F
        df['base|gender'] = df['base|gender'].map({0: 'M', 1: 'F'})
        
        first_map=df["MAP|MAP"][0]
        len_df=len(df)
        padding = np.zeros(len(df), dtype='bool')
        end_index=np.random.randint(start_len, len_df-pre_len)#为10
        if(end_index<max_lenth):
            true_data=df.iloc[0:end_index].copy()
            true_data = pd.concat([true_data,pd.DataFrame(index=range(max_lenth-end_index))], axis=0)#随机填充,增强对开始任务的学习能力量
            true_data = pd.concat([true_data,df.iloc[end_index:end_index+pre_len].copy()], axis=0)#随机填充,增强对开始任务的学习能力量

            selected_padding= np.copy(padding[0:end_index])
           
            selected_padding =  np.concatenate([ selected_padding,np.ones(max_lenth-end_index, dtype='bool')])

        else:
            true_data=df.iloc[end_index-max_lenth:end_index+pre_len].copy()

            selected_padding= np.copy(padding[end_index-max_lenth:end_index])
           


        
   

        key = true_data, self.max_seq_len, f'{self.gamma:.4f}', self.reward_dtype
        sample = self._get_sample(*key,map=first_map)
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


        if not self.brl_setting:
            print("error")
        else:
            reward = sample['reward'].astype('float32')


            cumreward = sample['cumreward'].astype('float32')


            action = (sample['action'][0].astype('int'), sample['action'][1].astype('int'))
            action = (action[0].clip(0, self.n_action), action[1].clip(0, self.n_action))#对action切片
            BIS_target= BIS_target.clip(0, n_bis)
            rp_target=rp_target.clip(0, n_rp)
            
            option = sample['option'].astype('int')
            option = option.clip(0, 4)
            mask_reward = bis > 0.01
            mask_reward[-1] = False
            

            action_prev = (np.roll(action[0], shift=1), np.roll(action[1], shift=1))  # 空出第一个位置
            action_prev[0][0] = 0
            action_prev[1][0] = 0
            action=np.copy(action_prev)


            obs=obs[0:max_lenth,:]
            

            if self.return_y:
                label_data = {
                    'aux': aux,
                    'mask_aux': mask_aux,
                    'reward': reward,
                    'cumreward': cumreward,
                    'action': action,
                    'mask_reward': mask_reward,
                    'bis_target': BIS_target,
                    'rp_target':rp_target,
                    'padding': selected_padding,
                }
                return (obs, action_prev, option, selected_padding), label_data  # 二维
            else:
                return (obs, action_prev, option, selected_padding)


class DataModule(pl.LightningDataModule):
    def __init__(self, df_path, task,batch_size, num_workers,#这是喂多少批
                 pin_memory, shuffle, data_args):
        super().__init__()
        self.df_path = df_path
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.data_args = data_args

    def setup(self, stage=None):
        data_kargs_default = self.data_args
        #取出df_path对应文件夹下的所有文件名
        csv_files = [file for file in os.listdir(self.df_path) if file.endswith('.csv')]
        # 假设 csv_files 是你要处理的 CSV 文件名列表
        csv_files_length = len(csv_files)
        percent_to_keep = rate_csv
        num_to_keep = int(csv_files_length * percent_to_keep)
        csv_files = csv_files[:num_to_keep]
        #按8:1分割csv_files
        n_train = int(len(csv_files) * train_r)
        train_list = csv_files[:n_train]
        valid_list = csv_files[n_train:]
        
         # 记录训练集和验证集的文件路径
        with open(os.path.join(self.df_path, '/data/tongqi/anesthesia/train_files.txt'), 'w') as f:
            for file in train_list:
                f.write(f"{file}\n")

        with open(os.path.join(self.df_path, '/data/tongqi/anesthesia/valid_files.txt'), 'w') as f:
            for file in valid_list:
                f.write(f"{file}\n")
        
        self.ds_train = TSRLDataset(train_list,**data_kargs_default)
        self.ds_valid = TSRLDataset(valid_list,**data_kargs_default)
        self.n_aux = self.ds_train.n_labels


        self.n_input = self.ds_train.n_features#获得一个病人的特征数
        self.max_length = self.ds_train.max_seq_len
        self.n_reward_max = self.ds_train.n_reward_max
        self.n_value_max = self.ds_train.n_value_max
        self.n_action = self.ds_train.n_action
        self.n_option = self.ds_train.n_option

    # not used
    def train_dataloader(self):  # 数据加载器在这里返回分割的数据
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False)

    def teardown(self, stage=None):
        pass
