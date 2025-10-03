import numpy as np
import pandas as pd

from functools import lru_cache
@lru_cache(10)
def _get_feat_meta(path):
    '''
    obtain all feature information
    and segment features based on continuous and discrete attributes
    '''
    if isinstance(path, pd.DataFrame):
        df_feat_meta = path
    else:
        df_feat_meta = pd.read_csv(path)
    df_meta_to_cont = df_feat_meta[df_feat_meta['key_type'] == 'cont']
    df_meta_to_cat = df_feat_meta[df_feat_meta['key_type'].str.contains('^cat')]
    return df_feat_meta, df_meta_to_cont, df_meta_to_cat

def onehot(df_sample,feat_meta_path):
    '''
            process features, expand discrete features with one hot
            '''
    feature_reindex = True
    # 依然是对特征值进行分类
    df_feat_meta, df_meta_to_cont, df_meta_to_cat = _get_feat_meta(feat_meta_path)

    df_case = df_sample
    feats = []
    # 找到cont对应的列,进行打印
    if feature_reindex:
        feats += [df_case.reindex(columns=df_meta_to_cont['feat_name'])]
    else:
        cols_cont = df_case.columns[df_case.columns.isin(df_meta_to_cont['feat_name'])]
        feats += [df_case.reindex(columns=cols_cont)]

    if feature_reindex:
        df_meta_to_cat_t = df_meta_to_cat
    else:
        df_meta_to_cat_t = df_meta_to_cat[df_meta_to_cat['feat_name'].isin(df_case.columns)]
    # 对cat属性进行遍历

    for _, sr_col in df_meta_to_cat_t.iterrows():
        col = sr_col['feat_name']
        vals = df_case[col]
        if not pd.isna(sr_col['cat']):  # 检查cat中是否为空,提取出mao,比如性别对应M,F
            cats = sr_col['cat'].split(',')
            cat2id = {c: i for i, c in enumerate(cats)}
            vals = vals.map(cat2id)
        # 离散变换,没看懂啥意思
        vals = pd.Categorical(vals, categories=np.arange(sr_col['n_dim']))
        vals = pd.get_dummies(vals)
        vals = vals.add_prefix(col + '_')
        feats += [vals]
    dim = 1
    df_case = pd.concat(feats, axis=dim)

    return df_case
def pip_fillna(df_sample, feat_meta_path):
    '''
    zero padding according to the continuous and discrete nature of the feature
    '''

    #根据keytype进行分类

    df_feat_meta, df_meta_to_cont, df_meta_to_cat = _get_feat_meta(feat_meta_path)


    df_case = df_sample
    #只获取take.columns中的列
    df_case = df_case.reindex(columns=df_feat_meta['feat_name'])
    #判断每一个位置缺失值
    df_mask = df_case.notna()

    for _, sr_col in df_feat_meta.iterrows():
        col = sr_col['feat_name']
        interp = sr_col['interp']
        feat_type = sr_col['key_type']
        val = df_case[col]
        val = val.fillna(0)
        df_case[col] = val
    return df_case, df_mask
def reward2return(rewards, gamma=0.9):
    rewards = rewards.fillna(0)
    result = np.zeros_like(rewards, dtype='float')
    steps = len(rewards)
    running_add = 0
    for i in reversed(range(steps)):
        running_add = running_add * gamma + rewards.values[i]
        result[i] = running_add
    result = pd.Series(result, index=rewards.index)
    return result
def cal_reward(df_case_raw, dtype='risk',f_map=map):
    c0, c1, c2 = 1.509, 1.084, 5.381
    # ["BIS", "MAP|MAP"]
    standard_map = f_map
    low_20_map = standard_map * 0.8
    low_10_map = standard_map * 0.9
    high_10_map = standard_map * 1.1
    high_20_map = standard_map * 1.2
    bis_low = 40
    bis_high = 60
    map = df_case_raw["MAP|MAP"].copy()
    bis = df_case_raw["BIS"].copy()
    # 计算
    risk_map = np.zeros_like(map)
    risk_map[map >high_20_map] =-1
    risk_map[map <=high_20_map] =0
    risk_map[map <=high_10_map] =1
    risk_map[map <=low_10_map] =0
    risk_map[map < low_20_map] = -1
    #如果map等于0,则返回0
    risk_map[map == 0] = 0
    #把bis中为0的值设置为1

    risk_bis =np.clip((10 * (c0 * (np.log((abs(bis-50)+50)*1.12) ** c1 - c2)) ** 2), a_min=0, a_max=7.75 * 2) / 7.75-1
    risk_bis[bis < bis_low] = -1
    risk_bis[bis==0] = 0

    risk_final = 0.2 * risk_map + 0.8 * risk_bis
    return risk_final


