from abc import ABC

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
# 仅在分布式训练的 rank=0 进程上执行被装饰的方法，避免重复打印
from pytorch_lightning.utilities.rank_zero  import rank_zero_only


class ConsoleLogger(Logger, ABC):
    """
    一个极简的控制台日志记录器，实现了 Lightning Logger 的核心接口：
    - 在控制台打印超参数与指标
    - 可选跟踪并输出某个监控指标（例如 'val_loss'）的当前最好值

    Attributes:
        monitor (str|None): 要监控的指标键名（例如 'val_loss'）。为 None 时不跟踪最佳值
        value (float|None): 记录到目前为止的最佳监控指标值；首次为 None
    """

    def __init__(self, monitor='val_loss'):
        super().__init__()
        self.monitor = monitor  # 注意：self.log_metrics() 要求 metrics 中必须包含该键，否则会抛出 KeyError
        # self.mode = mode
        self.value = None

    @property
    def name(self):
        """
        用于在 Lightning 内部标识该 logger 的名称

        Args:

        """
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @property
    @rank_zero_experiment
    def experiment(self):
        """
        兼容性属性：返回底层实验对象（如 TensorBoard 的 SummaryWriter）。
        本实现仅打印到控制台，因此返回 None。
        使用 @rank_zero_experiment 确保只在主进程访问。
        """
        return None

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        """
        log超参数配置。Lightning 会在合适时机调用该方法

        Args:
            params: 任意可序列化为字符串的超参数容器（如 dict、Namespace 等）
        """
        print(f'\nlog_hyperparams: {params}')

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: int) -> None:
        """
        记录并打印指标字典

        Args:
            metrics:
                - 必须包含键 'epoch'（用于打印当前轮次）
                - 其他键值对为待打印的标量指标（float 或可格式化为浮点）
                - 当设置了 monitor 时，metrics 必须包含该键
            step:
                - 当前训练中的全局 step（由 Trainer 维护）

        Behavior:
            - 打印：epoch、step 与各项指标（保留三位小数）
            - 若设置了 monitor，则以“更小更优”的准则跟踪并打印最优值
        """
        metrics = metrics.copy()  # 拷贝一份，避免对外部传入的字典产生副作用
        epoch = metrics.pop('epoch')
        metric_text = ', '.join(f'{k}: {v:.3f}' for k, v in metrics.items())
        print(f'  Metrics: epoch {epoch}, step {step}\n{{{metric_text}}}')

        # 如果配置了监控指标，按“更小更优”更新最佳值并打印
        if self.monitor is not None:
            value = metrics[self.monitor]
            if (self.value is None) or (value < self.value):
                self.value = value
                print(f' [Epoch {epoch}], [best {self.monitor}: {self.value:.3f}]')

    # not used
    @rank_zero_only
    def save(self) -> None:
        super().save()

    # not used
    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()
