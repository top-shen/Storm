import numpy as np
import pandas as pd
import torch
from typing import List, Union


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    return torch.divide(a, b, out=out, where=b != 0)


def _rank_ic_single(pred_row: np.ndarray, actual_row: np.ndarray) -> float:
    pred_row = np.asarray(pred_row, dtype=np.float64)
    actual_row = np.asarray(actual_row, dtype=np.float64)

    if pred_row.size <= 1 or actual_row.size <= 1:
        return 0.0
    if np.unique(pred_row).size < 2 or np.unique(actual_row).size < 2:
        return 0.0

    pred_rank = pd.Series(pred_row).rank(method="average").to_numpy(dtype=np.float64)
    actual_rank = pd.Series(actual_row).rank(method="average").to_numpy(dtype=np.float64)

    pred_centered = pred_rank - pred_rank.mean()
    actual_centered = actual_rank - actual_rank.mean()

    pred_std = pred_centered.std(ddof=0)
    actual_std = actual_centered.std(ddof=0)
    if pred_std <= 0 or actual_std <= 0:
        return 0.0

    covariance = np.mean(pred_centered * actual_centered)
    return float(covariance / (pred_std * actual_std))


def RankICSeries(preds: torch.Tensor, actuals: torch.Tensor):
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    else:
        preds = preds.reshape(preds.shape[0], -1)

    if actuals.dim() == 1:
        actuals = actuals.unsqueeze(0)
    else:
        actuals = actuals.reshape(actuals.shape[0], -1)

    preds_np = preds.detach().cpu().numpy()
    actuals_np = actuals.detach().cpu().numpy()
    values = [_rank_ic_single(pred_row, actual_row) for pred_row, actual_row in zip(preds_np, actuals_np)]
    return torch.tensor(values, device=preds.device, dtype=torch.float32)


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
    std_rank_ic = torch.std(values, unbiased=False)
    rank_ic_ir = _safe_div(mean_rank_ic, std_rank_ic)
    return rank_ic_ir


if __name__ == '__main__':
    preds = torch.tensor([0.3, 0.2, 0.9, 0.7, 0.1])
    actuals = torch.tensor([0.4, 0.1, 0.8, 0.6, 0.2])

    rank_ic = RankIC(preds, actuals)
    print(f"RankIC: {rank_ic.item()}")

    rank_ic_values = [rank_ic, rank_ic, rank_ic]
    rank_ic_ir = RankICIR(rank_ic_values)
    print(f"RankICIR: {rank_ic_ir.item()}")
