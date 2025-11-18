import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.utils import clone_module_list

import math
from typing import Optional, List

import torch
from torch import nn

from labml import tracker


class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## 准备多头注意力

    该模块执行线性变换并将向量分割为给定数量的头，用于多头注意力。
    这用于转换 **key**、**query** 和 **value** 向量。
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # 线性层用于线性变换
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # 头的数量
        self.heads = heads
        # 每个头中向量的维度
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # 输入形状 `[seq_len, batch_size, d_model]` 或 `[batch_size, d_model]`。
        # 我们对最后一个维度进行线性变换并将其分割为多个头。
        head_shape = x.shape[:-1]

        # 线性变换
        x = self.linear(x)

        # 将最后一个维度分割为头
        x = x.view(*head_shape, self.heads, self.d_k)

        # 输出形状 `[seq_len, batch_size, heads, d_k]` 或 `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a>

    ## 多头注意力模块

    该模块计算给定 `query`、`key` 和 `value` 向量的缩放多头注意力。

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    简而言之，它找到与查询匹配的键，并获取这些键的值。

    它使用查询和键的点积作为匹配程度的指标。
    在应用 $softmax$ 之前，点积被缩放为 $\frac{1}{\sqrt{d_k}}$。
    这样可以避免在 $d_k$ 很大时，点积值过大导致 softmax 给出非常小的梯度。

    Softmax 沿序列（或时间）轴计算。
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` 是头的数量。
        * `d_model` 是 `query`、`key` 和 `value` 向量中的特征数量。
        """

        super().__init__()

        # 每个头的特征数量
        self.d_k = d_model // heads
        # 头的数量
        self.heads = heads

        # 这些用于多头注意力的 `query`、`key` 和 `value` 向量的线性变换。
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # 沿时间维度的 softmax 注意力
        self.softmax = nn.Softmax(dim=1)

        # 输出层
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # softmax 前的缩放因子
        self.scale = 1 / math.sqrt(self.d_k)

        # 我们存储注意力以便于调试或其他计算
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### 计算查询和键之间的分数

        该方法可以被重写以实现其他变体，如相对注意力。
        """

        # 计算 $Q K^\top$ 或 $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        `mask` 形状为 `[seq_len_q, seq_len_k, batch_size]`，其中第一维是查询维度。
        如果查询维度等于 $1$，则会广播。
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # 相同的掩码应用于所有头。
        mask = mask.unsqueeze(-1)

        # 结果掩码形状 `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`、`key` 和 `value` 是存储
        一组 *query*、*key* 和 *value* 向量的张量。
        它们的形状为 `[seq_len, batch_size, d_model]`。

        `mask` 形状为 `[seq_len, seq_len, batch_size]`，
        `mask[i, j, b]` 表示对于批量 `b`，
        位置 `i` 的查询是否可以访问位置 `j` 的键和值。
        """

        # `query`、`key` 和 `value` 形状 `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # 准备 `query`、`key` 和 `value` 以进行注意力计算。
        # 这些现在形状为 `[seq_len, batch_size, heads, d_k]`。
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # 计算注意力分数 $Q K^\top$。
        # 结果张量形状 `[seq_len, seq_len, batch_size, heads]`。
        scores = self.get_scores(query, key)

        # 缩放分数 $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 沿键序列维度的 $softmax$
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # 保存注意力以便于调试
        tracker.debug('attn', attn)

        # 应用 dropout
        attn = self.dropout(attn)

        # 乘以值
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # 保存注意力用于其他计算
        self.attn = attn.detach()

        # 拼接多个头
        x = x.reshape(seq_len, batch_size, -1)

        # 输出层
        return self.output(x)


class SwitchFeedForward(Module):
    """
    ## 在多个 FFN 之间进行路由
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: FeedForward,
                 d_model: int):
        """
        * `capacity_factor` 是每个专家的容量，相对于理想负载的因子
        * `drop_tokens` 指定是否在路由到某个专家的 token 超过容量时丢弃这些 token
        * `is_scale_prob` 指定是否将输入乘以路由概率
        * `n_experts` 是专家的数量
        * `expert` 是专家层，一个 [FFN 模块](../feed_forward.html)
        * `d_model` 是 token 嵌入的特征数量
        * `d_ff` 是 FFN 隐藏层的特征数量
        * `dropout` 是 FFN 中的 dropout 概率
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # 创建多个 FFN 的副本
        self.experts = clone_module_list(expert, n_experts)
        # 路由层和 softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` 是传递给切换模块的输入，形状 `[seq_len, batch_size, d_model]`
        """

        # 捕获形状以便后续更改形状
        seq_len, batch_size, d_model = x.shape
        # 将序列和批量维度展平
        x = x.view(-1, d_model)

        # 获取每个 token 的路由概率。
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # 其中 $N$ 是专家数量 `n_experts`，
        # $h(\cdot)$ 是 token 嵌入的线性变换。
        route_prob = self.softmax(self.switch(x))

        # 获取最大路由概率和路由路径。
        # 我们路由到具有最高概率的专家
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # 获取路由到每个专家的 token 索引
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # 初始化一个空张量来存储输出
        final_output = x.new_zeros(x.shape)

        # 每个专家的容量。
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # 路由到每个专家的 token 数量
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # 初始化一个空列表以存储被丢弃的 tokens
        dropped = []
        # 仅在 `drop_tokens` 为 `True` 时丢弃 tokens
        if self.drop_tokens:
            # 在每个专家中丢弃 tokens
            for i in range(self.n_experts):
                # 如果专家未超出容量，则忽略
                if len(indexes_list[i]) <= capacity:
                    continue
                # 在丢弃前随机打乱索引
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # 收集超过容量的 token 作为丢弃的 token
                dropped.append(indexes_list[i][capacity:])
                # 仅保留容量范围内的 token
                indexes_list[i] = indexes_list[i][:capacity]

        # 获取专家 FFN 的输出
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # 分配到最终输出
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i].to(final_output.dtype)

        # 处理被丢弃的 tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # 将专家输出乘以概率 $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # 不缩放值，但乘以 $\frac{p}{\hat{p}} = 1$ 以使梯度流动
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # 将最终输出形状改回 `[seq_len, batch_size, d_model]`
        final_output = final_output.view(seq_len, batch_size, d_model)

        # 返回
        #
        # * 最终输出
        # * 路由到每个专家的 token 数量
        # * 每个专家的概率总和
        # * 被丢弃的 token 数量
        # * 选择的专家的路由概率
        #
        # 这些用于负载平衡损失和日志记录
        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


class SwitchTransformerLayer(Module):
    """
    # Switch Transformer 层
    """

    def __init__(self, *, d_model: int, attn: MultiHeadAttention, feed_forward: SwitchFeedForward, dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *, x: torch.Tensor, mask: torch.Tensor, additional_features: Optional[torch.Tensor] = None):
        """
        参数：
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [seq_len, seq_len, batch_size]
            additional_features: 已投影的外部特征 [batch_size, seq_len, d_model]
        """
        # 在自注意力前进行归一化
        z = self.norm_self_attn(x)  # [batch_size, seq_len, d_model]
        
        # 转置为 [seq_len, batch_size, d_model] 以适应 MHA
        z = z.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # 运行自注意力
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)  # [seq_len, batch_size, d_model]
        
        # 转置回 [batch_size, seq_len, d_model]
        self_attn = self.attn(query=z, key=z, value=z, mask=mask).transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 如果提供了外部特征，进行加法操作
        if additional_features is not None:
            x = x + self.dropout(self_attn) + additional_features
        else:
            x = x + self.dropout(self_attn)

        # 在前馈网络前进行归一化
        z = self.norm_ff(x)  # [batch_size, seq_len, d_model]
        
        # 通过前馈网络
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        
        # 添加前馈网络的输出
        x = x + self.dropout(ff)

        return x, counts, route_prob, n_dropped, route_prob_max


class SwitchTransformer(Module):
    """
    ## Switch Transformer
    """

    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        # 创建多个 Transformer 层的副本
        self.layers = clone_module_list(layer, n_layers)
        # 最终归一化层
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, additional_features: Optional[torch.Tensor] = None):
        """
        参数：
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [seq_len, seq_len, batch_size]
            additional_features: 已投影的外部特征 [batch_size, seq_len, d_model]
        """
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []

        for i, layer in enumerate(self.layers):
            if i == 0 and additional_features is not None:
            #if additional_features is not None:
                # 第一层使用 pooled_features
                x, f, p, n_d, p_max = layer(x=x, mask=mask, additional_features=additional_features)
            else:
                x, f, p, n_d, p_max = layer(x=x, mask=mask)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)

        # 最终归一化
        x = self.norm(x)
        return x, torch.stack(counts), torch.stack(route_prob), n_dropped, torch.stack(route_prob_max)

