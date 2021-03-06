import torch
import numpy as np
import torch.nn as nn
import copy


def corrcoef(x, rowvar=True):
    """
    code from
    https://github.com/AllenCellModeling/pytorch_integrated_cell/blob/8a83fc6f8dc79037f4b681d9d7ef0bc5b91e9948/integrated_cell/corr_stats.py
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor
    rowvar : bool, default True means every single row is a variable, and every single column is an observation, e.g. a sample
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    # 计算每个变量的均值，默认每行是一个变量，每列是一个sample
    if not rowvar and len(x.size()) != 1:
        x = x.T
    mean_x = torch.mean(x, 1).unsqueeze(1)
    # xm(j, i)是第i个sample的第j个变量，已经被减去了j变量的均值，等于论文中的F(si)j- uj,
    # xm(k, i)是第i个sample的第k个变量，已经被减去了k变量的均值，等于论文中的F(si)k- uk,
    xm = x.sub(mean_x.expand_as(x))
    # c(j, k) 等于论文中 M(j, k)的分子, c也是F(s)的协方差矩阵Cov(F(s), F(s))
    c = xm.mm(xm.t())
    # 协方差矩阵一般会除以 num_sample - 1
    # c = c / (x.size(1) - 1)

    # normalize covariance matrix
    # dj是每个变量的方差, E[(F(s)j - uj)^2]，也即j == k 时的分子
    d = torch.diag(c)
    # 取标准差
    stddev = torch.pow(d + 1e-7, 0.5)  # 防止出现0，导致nan
    # 论文中除以的分母
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def disentangling_loss(feature):
    # feature = [batch_size, hid_dim]
    M = corrcoef(feature, rowvar=False)
    # M = [hid_dim, hid_dim]
    loss_decorr = 0.5 * (torch.sum(torch.pow(M, 2)) - torch.sum(torch.pow(torch.diag(M), 2)))
    return loss_decorr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    else:
        for param in module.parameters():
            nn.init.uniform_(param, -0.02, 0.02)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_metrics(metrics, mode=''):
    # TODO 把print metrics统一起来
    print("#" * 20, mode + ' metrics ', "#" * 20)
    for k, v in metrics.items():
        print(f'\t{k}: {v:.5f}')
    print("#" * 20, ' end ', "#" * (24 + len(mode)))


def write_metrics(metrics, file, mode=''):
    file.write("#" * 20 + mode + ' metrics ' + "#" * 20 + '\n')
    for k, v in metrics.items():
        file.write(f'\t{k}: {v:.5f}\n')
    file.write("#" * 20 + ' end ' + "#" * 20 + '\n')


def write_metrics_to_writer(metrics, writer, global_step, mode=''):
    for k, v in metrics.items():
        writer.add_scalar(f'{mode}_{k}', v, global_step)


def clones(module, N):
    """生成N个同样的layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """将后续部分mask掉"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
