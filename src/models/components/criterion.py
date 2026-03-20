import torch
from torch import nn


class MSECrit(nn.Module):
    def __init__(self, mse_weight=1.0) -> None:
        super().__init__()
        self.mse_weight = mse_weight

    def forward(self, outputs, target, prefix="", weight=1, mask=None) -> torch.Tensor:
        error_dict = {}
        diff = outputs - target
        if mask is None:
            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
        else:
            mse = (diff**2 * mask).sum() / mask.sum()
            mae = (diff.abs() * mask).sum() / mask.sum()
        rmse = torch.sqrt(mse)

        error_dict[prefix + "_mae"] = mae
        error_dict[prefix + "_rmse"] = rmse
        error_dict[prefix + "_loss"] = mse * self.mse_weight * weight
        return error_dict


class MixedCrit(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def forward(self, outputs, target, prefix="", weight=1, mask=None):
        error_dict = {}
        diff = outputs - target
        if mask is None:
            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
        else:
            mse = (diff**2 * mask).sum() / mask.sum()
            mae = (diff.abs() * mask).sum() / mask.sum()
        rmse = torch.sqrt(mse)

        error_dict[prefix + "_mae"] = mae
        error_dict[prefix + "_rmse"] = rmse
        error_dict[prefix + "_loss"] = (mse * self.lam + mae) * weight
        return error_dict
