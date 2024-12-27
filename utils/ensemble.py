import CRPS.CRPS as pscore
import torch
from torch.distributions.normal import Normal


def crps_cost_function(y_true, mu, sigma):
    """
    计算CRPS代价函数

    参数:
    y_true (torch.Tensor): 真实的观测值，形状可以是 (batch_size, *)，*表示可能的其他维度（比如多个样本时）
    mu (torch.Tensor): 预测的正态分布均值，形状需与y_true匹配，(batch_size, *)
    sigma (torch.Tensor): 预测的正态分布标准差，形状需与y_true匹配，(batch_size, *)

    返回:
    torch.Tensor: CRPS代价函数的值，形状为 (batch_size, *)，可用于反向传播进行优化
    """
    # 创建正态分布对象
    dist = Normal(mu, sigma)
    # 生成用于积分的点，这里简单地在 [mu - 3*sigma, mu + 3*sigma] 区间内均匀采样，可根据实际调整点数和范围
    integration_points = mu.unsqueeze(-1) + torch.linspace(-3, 3, 100).to(mu.device).unsqueeze(0).unsqueeze(0) * sigma.unsqueeze(-1)
    # 计算预测分布在这些积分点上的累积分布函数 (CDF)
    cdf_pred = dist.cdf(integration_points)
    # 计算真实观测值对应的阶跃函数，broadcast到和cdf_pred相同的形状
    step_function = (integration_points < y_true.unsqueeze(-1)).float()
    # 计算累积分布函数与阶跃函数之间的差值
    diff_cdf = cdf_pred - step_function
    # 计算差值的平方
    diff_cdf_squared = diff_cdf ** 2
    # 沿着积分点维度求平均，得到每个样本的CRPS值（可用于反向传播）
    crps_value = torch.mean(diff_cdf_squared, dim=-1)
    return crps_value

def CRPSmetric(output: torch.tensor, groundtruth:torch.tensor) -> torch.tensor:
    # param: output(1, patch_len, ensemble_num)
    # param: groundtruth(1, patch_len, 1)
    res_crps = torch.zeros_like(groundtruth)
    B,P,E = output.shape# batch_size, patch_len, ensemble_num
    output = output.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    for i in range(B):
        for j in range(P):
            res_crps[i,j] = torch.tensor(pscore(output[i,j], groundtruth[i,j]).compute()[0])
    return res_crps
