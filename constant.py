"""相关文件路径"""
# path_check_point='anesthesia/output/rlditr/log/lightning_logs/'+'version_2/'+'checkpoint/last.ckpt'
df_path = '/data/tongqi/anesthesia/DATA/TRAINDATA_BBF_RFTN_ASA_DIAGNOSE'  # 数据路径,包含每一个人
raw_p = '/data/tongqi/anesthesia/'  # 项目根目录
model_dir = raw_p + 'output/rlditr/'  # 模型读取路径
output_dir = raw_p + 'output/rlditr'  # 模型保存路径

max_lenth = 64  # 单场手术最大长度
start_len = 10
pre_len = 5  # 预测步数
mode=1  # 检查模式
sample_train = False  # 采样训练/贪心
n_hidden=256  # Transformer的嵌入维数，同时也是PE的维数，注意PE是拼接
batch_size=128
gpus = 0
rate_csv=1
n_bis = 100
n_rp = 200
num_workers=0
pin_memory=True
task = 'rlsl'

shuffle = True
lr = 0.0001
n_epoch = 1000
patience = n_epoch
data_kargs_default = {
            'df_meta_path': raw_p+'DATA/task.columns.csv',
            'max_seq_len': max_lenth,  # 这是一批的最大长度
            'brl_setting': True,
            'reward_dtype': 'risk',
            'return_y': True,
            'feat_append_mask': True,
        }
add_mask=False
col_action=['BBF_SPEED', 'RFTN_SPEED']
option='PERFORM_OPTION'
option_map = {
            'test': 1
}
n_reward_max=1
n_value_max=64  # return的最大值
gamma=0.9  # 贴现因子
n_action = 200+1  # 注射速率取值，连续值不考虑，值为[0,200]
n_option = 2#只考虑一个option,就是只注射
# 磁盘缓存设置
cache_dir= 'chache'  # 缓存目录
cache_renew=True
diskCache = None
cols_state=["BIS","MAP|MAP"]
cols_aux=["MAP|MAP","MAP|AP_sys","MAP|AP_dia"]
train_r=0.9

nhead=8
nhid=2048
nlayers=3
dropout: float = 0.4
#val
#pre
val_rate=1

model_val=1

# BC+MCTS混合训练配置
use_mcts = True  # 是否使用MCTS搜索
lambda_bc = 0.7  # BC损失权重（模仿专家）
lambda_mcts = 0.3  # MCTS损失权重（学习MCTS找到的高价值动作）
teacher_forcing = True  # Stage 2是否使用专家动作rollout（推荐True，更稳定）

# MCTS搜索参数
mcts_simulations = 50  # MCTS每次搜索的模拟次数
mcts_c_puct = 1.0  # MCTS的探索常数（UCB）
use_mcts_margin = 0.1  # MCTS动作需要比专家好多少才使用（margin）

# 批量MCTS配置
use_mcts_batch_search = False  # True=对所有样本MCTS（慢但完整），False=只对部分样本（快）
mcts_batch_samples = 32  # 快速模式下，每个batch最多对多少样本执行MCTS

# 训练策略配置
warmup_epochs = 10  # 前N个epoch只训练环境模型，不训练policy
env_loss_weight = 2.0  # warmup后，环境模型loss的权重（相对policy）

# 其他选项（预留）
use_scheduled_sampling = False  # 是否使用scheduled sampling
scheduled_sampling_decay = 0.01  # scheduled sampling的衰减率（每epoch）

