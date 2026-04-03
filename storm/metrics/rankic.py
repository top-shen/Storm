import torch
from typing import List, Union


def RankICSeries(preds: torch.Tensor, actuals: torch.Tensor):

    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    else:
        preds = preds.reshape(preds.shape[0], -1)

    if actuals.dim() == 1:
        actuals = actuals.unsqueeze(0)
    else:
        actuals = actuals.reshape(actuals.shape[0], -1)

    preds_rank = preds.argsort(dim=1).argsort(dim=1).float()
    actuals_rank = actuals.argsort(dim=1).argsort(dim=1).float()

    preds_centered = preds_rank - preds_rank.mean(dim=1, keepdim=True)
    actuals_centered = actuals_rank - actuals_rank.mean(dim=1, keepdim=True)

    covariance = torch.mean(preds_centered * actuals_centered, dim=1)
    preds_std = preds_rank.std(dim=1)
    actuals_std = actuals_rank.std(dim=1)

    rank_ic_series = covariance / (preds_std * actuals_std)
    return rank_ic_series


def RankIC(preds: torch.Tensor, actuals: torch.Tensor):
    return RankICSeries(preds, actuals).mean()


def RankICIR(rank_ic_values: Union[List[torch.Tensor], torch.Tensor]):

    if isinstance(rank_ic_values, torch.Tensor):
        values = rank_ic_values.reshape(-1).float()
    else:
        if len(rank_ic_values) == 0:
            return torch.tensor(0.0)

        tensors = []
        device = None
        for value in rank_ic_values:
            if torch.is_tensor(value):
                device = value.device
                break
        if device is None:
            device = torch.device("cpu")

        for value in rank_ic_values:
            if torch.is_tensor(value):
                tensors.append(value.reshape(-1).float())
            else:
                tensors.append(torch.tensor([value], device=device, dtype=torch.float32))

        values = torch.cat(tensors, dim=0)

    if values.numel() <= 1:
        return torch.tensor(0.0, device=values.device)

    mean_rank_ic = torch.mean(values)
    std_rank_ic = torch.std(values)

    rank_ic_ir = mean_rank_ic / std_rank_ic
    return rank_ic_ir


if __name__ == '__main__':
    preds = torch.tensor([0.3, 0.2, 0.9, 0.7, 0.1])
    actuals = torch.tensor([0.4, 0.1, 0.8, 0.6, 0.2])

    rank_ic = RankIC(preds, actuals)
    print(f"RankIC: {rank_ic.item()}")

    rank_ic_values = [rank_ic, rank_ic, rank_ic]
    rank_ic_ir = RankICIR(rank_ic_values)
    print(f"RankICIR: {rank_ic_ir.item()}")