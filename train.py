#!/usr/bin/env python3

import fire
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from RL.Module.data import DataModule
from light.core.logger import ConsoleLogger
from pytorch_lightning.loggers import CSVLogger
from constant import *
from RL.Module.model import TransformerPlanningModel
from RL.Module.pl_model import RLSLModelModule

# 使用 Tensor Cores 计算 float32 matmul，精度接近 FP32，速度常显著提升
torch.set_float32_matmul_precision('high')  # highest/high/medium

def get_model(data_module=None):
    n_input = data_module.n_input
    max_length = data_module.max_length
    n_reward_max = data_module.n_reward_max
    n_value_max = data_module.n_value_max
    n_action = data_module.n_action
    n_option = data_module.n_option
    n_aux = data_module.n_aux


    model_kargs_ori = {}
    model_kargs_ori.update({
        'n_input': n_input,
        'max_lenth': max_length,
        'n_action': n_action,
        'n_option': n_option,
        'n_reward_max': n_reward_max,
        'n_value_max': n_value_max,
        'n_aux': n_aux,
    })

    model = TransformerPlanningModel(**model_kargs_ori)
    return model


class LightRunner(object):
    # 划分验证集的比例, 用于训练的GPU的数量, pin_memory, 写回内存
    def __init__(self):  # num_workers是用于数据加载的线程数
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._gpus = gpus
        self._shuffle = shuffle
        self._n_epoch = n_epoch
        self._patience = patience
        self._lr = lr
        self._task = task
        self._df_path = df_path
        self._output_dir = output_dir
        self._max_lenth = max_lenth
        self._data_kargs_default = data_kargs_default

    def train(self):
        # 获取可见 GPU 设备数量
        num_gpus = torch.cuda.device_count()

        # 遍历每个 GPU，打印其内存占用情况
        for gpu_id in range(num_gpus):
            torch.cuda.device(gpu_id)
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            print(f"GPU {gpu_id}: Allocated memory: {allocated_memory / (1024 * 1024)} MiB, Reserved memory: {reserved_memory / (1024 * 1024)} MiB")

        torch.cuda.empty_cache()  # 清空未使用的缓存内存
        output_dir = Path(self._output_dir)
        # data module
        data_module = DataModule(df_path=self._df_path, task=task,
                                 batch_size=self._batch_size, num_workers=self._num_workers,
                                 pin_memory=self._pin_memory, shuffle=shuffle,data_args=self._data_kargs_default)

        data_module.setup()
        print('data finish')
        model = get_model(data_module=data_module)
        model_kargs = dict(model=model, lr=lr,
                           output_dir=output_dir,
                           data_module=data_module)

        if task == 'rlsl':
            
            model_module = RLSLModelModule(**model_kargs)
   
            # trainer
            log_dir = output_dir / 'log'
            logger_csv = CSVLogger(save_dir=str(log_dir), name='lightning_logs')
            version = logger_csv.version
            logger_tb = TensorBoardLogger(save_dir=str(log_dir), name='tb', version=version)
            version_dir = Path(logger_csv.log_dir)  # 版本号自动递增

            trainer = Trainer(
                accelerator="gpu",
                devices=[gpus],
                max_epochs=n_epoch,
                logger=[
                    ConsoleLogger(),  # 控制台
                    logger_csv,  # CSV文件
                    logger_tb,  # TensorBoard
                ],
                callbacks=[
                    # 保存验证集上表现最好的模型
                    ModelCheckpoint(dirpath=(version_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                                    monitor="val_loss", mode="min", save_top_k=1),
                    # 显示训练进度条
                    TQDMProgressBar(refresh_rate=10),
                ],
                strategy='auto',  # 自动选择分布式训练策略
                val_check_interval=1.0,
                # use (float) to check within a training epoch：此时这个值为一个epoch的百分比。每百分之多少测试一次。use (int) to check every n steps (batches)：每多少个batch测试一次。
            )
            if mode==1:
                trainer.fit(model_module, datamodule=data_module)
            # else: 
            #     trainer.fit(model_module, datamodule=data_module,ckpt_path=path_check_point)
          

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            # save model
            print('train over')
            if trainer.global_rank == 0:
                model = model_module.model
                torch.save(model.state_dict(), str(version_dir / 'state_dict.zip'))
                torch.save(model, str(version_dir / 'model.pt'))
                print('model saved')



if __name__ == '__main__':
    fire.Fire(LightRunner)
