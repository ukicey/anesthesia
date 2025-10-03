import math
from typing import Optional

import torch
from torch import nn as nn

# 提供一些基本模块

class PositionalEncoding(nn.Module):
    """
    预制 Transformer 论文中的正弦位置编码

    Attributes:
        d_model: 模型嵌入维数
        max_len: 最大序列长度
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 把预计算好的位置编码 pe 注册为缓冲区，避免当成可训练权重，又保证保存与设备迁移正常
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        """
        提取与输入张量 x 形状匹配的正弦位置编码

        Args:
            x: 输入张量
            offset: 位置起点偏移量

        Returns:
            对应输入长度的正弦位置编码张量，形状与输入除最后一维外保持一致
            最后一维为位置编码维度
        """
        length = x.shape[1]
        shape = list(x.shape)
        shape[-1] = self.pe.shape[-1]

        # None 用来加一个维度，然后扩展成输入 x 的形状
        return self.pe[None, offset:offset + length].expand(shape)

# 基于Transformer的因果编码器，用于处理序列数据，在初始推理时使用一次
class TransformerCausalEncoder(nn.Module):
    """
    实现一个基于Transformer的编码器。这个编码器的作用是处理输入序列X，生成相应的输出序列Y，并且在处理过程中引入了一定的时间顺序关系

    Attributes:
        n_input:
        n_hidden: 隐层嵌入维数，同时也是PE维数，这么做实现维度对齐自由（PE为拼接而非点加）
        nhead:
        nhid: FNN中间隐层维数
        nlayers:
        max_len:
        dropout: 随机失活率
    """
    def  __init__(self, n_input, n_hidden, nhead, nhid, nlayers, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.model_type = 'Transformer'
        self.n_hidden = n_hidden
        self.nhid = nhid
        self.n_input = n_input
        self.max_len = max_len
        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout
        self.forward_mask = nn.Parameter(self.generate_square_subsequent_mask(max_len * 2), requires_grad=False)  # 有长度冗余

        self.pos_encoder = PositionalEncoding(n_hidden)  # PE维数 = n_hidden
        self.input_encoder = nn.Sequential(
            # nn.Linear(817, n_hidden),
            nn.Linear(self.n_input + self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.LayerNorm(self.n_hidden),
        )
        encoder_layers = nn.TransformerEncoderLayer(self.n_hidden, self.nhead, self.nhid, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)

    @staticmethod
    def generate_square_subsequent_mask(size: int) -> torch.Tensor:
        """
        生成一个因果掩码矩阵，确保每个位置只能关注其之前的位置

        Args:
            size: 矩阵大小

        Returns:

        """
        m = torch.full((size, size), float('-inf'),)
        m = torch.triu(m, diagonal=1)
        return m


    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        """

        Args:
            x: 输入张量 (B, L, E)
            padding_mask: 填充掩码 (B, L) (bool)
                - True 表示该位置被忽略
            dynamics: 是否动态构造掩码
        """
        b, l, d = x.shape  # b: batch size, l: sequence length, d: feature dimension
        x = torch.cat([x, self.pos_encoder(x)], dim=-1)  # 拼接位置编码
        x = self.input_encoder(x)
        x = x.transpose(0, 1)  # nn.TransformerEncoder（默认 batch_first=False）期望输入张量形状为 (L, B, E)

  
        if dynamics:
            forward_mask = self.generate_square_subsequent_mask(l)
        else:
            forward_mask = self.forward_mask[:l, :l]
        
        feature = self.transformer_encoder(x, forward_mask, padding_mask)
        feature = feature.transpose(0, 1)
        return feature  # (B, L, E)

# 基于Transformer的因果解码器，用于处理序列数据
class TransformerCausalDecoder(nn.Module):
    def __init__(self, n_input, n_hidden=256, nhead=8, nhid=2048, nlayers=3, max_len: int = 512, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.nhid = nhid
        self.max_len = max_len
        self.dropout = dropout
        self.nlayers = nlayers
        self.nhead = nhead
        self.forward_mask = nn.Parameter(self.generate_square_subsequent_mask(self.max_len * 2), requires_grad=False)

        self.pos_encoder = PositionalEncoding(n_hidden)  # PE维数 = n_hidden
        self.input_encoder = nn.Sequential(
            # nn.Linear(1025, n_hidden),
            nn.Linear(self.n_input + self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.LayerNorm(self.n_hidden),
        )

        decoder_layers = nn.TransformerDecoderLayer(self.n_hidden, self.nhead, self.nhid, self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.nlayers)

    @staticmethod
    def generate_square_subsequent_mask(size: int) -> torch.Tensor:
        """
        生成一个因果掩码矩阵，确保每个位置只能关注其之前的位置

        Args:
            size: 矩阵大小

        Returns:

        """
        m = torch.full((size, size), float('-inf'), )
        m = torch.triu(m, diagonal=1)
        return m


    def forward(self, q: torch.Tensor, v: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                dynamics: bool = False, offset=0):
        """
        X: (B,L,E)
        x_mask: (B,L,E) (bool)
            - imputation and padding
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        NOTE:
        - transformer_encoder input
            - src (S,N,E)
            - src_mask (S,S)
            - src_key_padding_mask (N,S)
        """
        b, l, d = q.shape
        q = torch.cat([q, self.pos_encoder(q, offset=offset-1)], dim=-1)  # 拼接
        q = self.input_encoder(q)  # feature
        q = q.transpose(0, 1)  # 矩阵的行和列会互换位置
        v = v.transpose(0, 1)

        if dynamics:
            forward_mask = self.generate_square_subsequent_mask(l)
        else:
            forward_mask = self.forward_mask[:l, :l]

        # feat = self.transformer_decoder(q, v, tgt_mask=forward_mask, memory_mask=forward_mask,
        #                                 memory_key_padding_mask=padding_mask, tgt_key_padding_mask=padding_mask)

        # 这里是不是不该给 memory 加 mask？
        feat = self.transformer_decoder(q, v, tgt_mask=forward_mask, memory_mask=forward_mask,
                                        memory_key_padding_mask=padding_mask, tgt_key_padding_mask=padding_mask)

        feat = feat.transpose(0, 1)
        return feat  # (B, L, E)


class MLP(nn.Module):
    """
    Attributes:
        input_size: 输入维数
        layer_sizes (list[int]): 隐层维数列表
        output_size: 输出维数
        output_activation: 输出层激活函数
        activation: 隐层激活函数
    """
    def __init__(self, input_size, layer_sizes:list[int], output_size, output_activation=torch.nn.Identity, activation=torch.nn.ELU):
        super().__init__()

        sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            act = output_activation if i == len(sizes) - 2 else activation
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
        self.mlp = torch.nn.Sequential(*layers)

    # kvarg: padding_mask: Optional[torch.Tensor] = None
    def forward(self, x,):
        return self.mlp(x)

# # not used
# class Transpose(nn.Module):
#     def __init__(self, dim0, dim1):
#         super().__init__()
#         self.dim0 = dim0
#         self.dim1 = dim1
#
#     def forward(self, x):
#         """
#         在输入张量 x 上执行转置操作
#
#         Args:
#             x: 输入张量
#
#         Returns:
#             转置后的张量
#         """
#         return x.transpose(self.dim0, self.dim1)
#
# # not used
# class CNNCausalBlock(nn.Module):
#     def __init__(self, input_size, output_size, activation=torch.nn.ELU):
#         super().__init__()
#         self.cnn = torch.nn.Conv1d(input_size, output_size, kernel_size=3, padding=2)
#         self.act = activation()
#
#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.act(x)
#         x = x[..., :-2]
#         return x
#
# # not used
# class CNN(nn.Module):
#     def __init__(self, input_size, layer_sizes, output_size,
#                  output_activation=torch.nn.Identity, activation=torch.nn.ELU):
#         super().__init__()
#
#         sizes = [input_size] + layer_sizes + [output_size]
#         layers = []
#         for i in range(len(sizes) - 1):
#             act = activation if i < len(sizes) - 2 else output_activation
#             layers += [CNNCausalBlock(sizes[i], sizes[i + 1], act)]
#         self.layers = torch.nn.Sequential(*layers)
#
#     def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
#         x = x.transpose(1, 2)
#         y = self.layers(x)
#         y = y.transpose(1, 2)
#         return y
#
# # not used
# class GEGLU(nn.Module):
#     def forward(self, x):
#         x, gates = x.chunk(2, dim=-1)
#         return x * F.gelu(gates)
#
# # not used
# class FeedForward(nn.Module):
#     def __init__(
#             self,
#             dim,
#             mult=4,
#             dropout=0.
#     ):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim * mult * 2),
#             GEGLU(),  # 嵌入维数减半
#             nn.Dropout(dropout),
#             nn.Linear(dim * mult, dim)
#         )
#
#     def forward(self, x):
#         x = self.norm(x)  # 前置归一化
#         return self.net(x)
#
# # not used
# class CausalMultiHeadAttn(nn.Module):
#     '''
#     p(y|x) model
#
#     X[1:T] -> Y[1:T]
#
#     X[1:t]+Y[1:t]+Y[t+1:T] -> Y[t+1:T]
#
#     norm is placed behind the active layer https://www.zhihu.com/question/283715823
#     '''
#
#     def __init__(self, n_input, n_hidden, nhead, max_len: int = 512, dropout: float = 0.0):
#         super().__init__()
#         self.model_type = 'Transformer'
#         self.n_hidden = n_hidden
#         self.max_len = max_len
#         self.forward_mask = nn.Parameter(self.generate_square_subsequent_mask(max_len * 2), requires_grad=False)
#         self.pos_encoder = PositionalEncoding(n_hidden)
#
#         self.input_encoder = nn.Sequential(
#             nn.Linear(n_input + n_hidden, n_hidden),
#             nn.ReLU(),
#             nn.LayerNorm(n_hidden),
#         )
#         self.transformer_encoder = nn.MultiheadAttention(embed_dim=n_input, num_heads=nhead, batch_first=True)
#
#     def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
#         # src_mask: If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def forward(self, q: torch.Tensor, v: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
#                 dynamics: bool = False):
#         '''
#         X: (B,L,E)
#         x_mask: (B,L,E) (bool)
#             - imputation and padding
#         padding_mask: (B,L) (bool)
#             - True Non zero means ignored
#             - bool tensor
#             - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
#
#         NOTE:
#         - transformer_encoder input
#             - src (S,N,E)
#             - src_mask (S,S)
#             - src_key_padding_mask (N,S)
#         '''
#         b, l, d = v.shape
#         v = torch.cat([v, self.pos_encoder(v)], dim=-1)
#         v = self.input_encoder(v)
#         if dynamics:
#             forward_mask = self.generate_square_subsequent_mask(l)
#         else:
#             forward_mask = self.forward_mask[:l, :l]
#
#         feat, attn_output_weights = self.transformer_encoder(query=q, key=v, value=v, key_padding_mask=padding_mask,
#                                                              attn_mask=forward_mask)
#         return feat
#
# # not used
# class TransformerModel(nn.Module):
#     '''
#     X[1:T] -> Y[1:T]
#     X[1:t]+Y[1:t]+Y[t+1:T] -> Y[t+1:T]
#     '''
#
#     def __init__(self, n_outputs, n_input, n_hidden=256, nhead=8, nhid=2048, nlayers=3,
#                  max_len=512, dropout: float = 0.5):
#         super().__init__()
#         self.model_type = 'Transformer'
#         self.n_hidden = n_hidden
#         self.nhid = nhid
#         self.max_len = max_len
#         self.encoder = TransformerCausalEncoder(n_input, n_hidden, nhead, nhid, nlayers, max_len=max_len,
#                                                 dropout=dropout)
#         self.decoder = nn.Linear(n_hidden, n_outputs)
#
#     def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
#         '''
#         X: (B,L,E)
#         x_mask: (B,L,E) (bool)
#             - imputation and padding
#         padding_mask: (B,L) (bool)
#             - True Non zero means ignored
#             - bool tensor
#             - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
#
#         NOTE:
#         - transformer_encoder input
#             - src (S,N,E)
#             - src_mask (S,S)
#             - src_key_padding_mask (N,S)
#         '''
#
#         feat = self.encoder(x, padding_mask=padding_mask)
#         output = self.decoder(feat)
#         return output


if __name__ == '__main__':
    pass