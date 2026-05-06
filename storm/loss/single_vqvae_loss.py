import torch
from torch import nn
import torch.nn.functional as F

from storm.registry import LOSS_FUNC

@LOSS_FUNC.register_module(force=True)
class SingleVQVAELoss(nn.Module):
    def __init__(self,
                 cs_scale = 1.0,
                 nll_loss_weight = 1.0,
                 ret_loss_weight = 1.0,
                 kl_loss_weight = 0.000001,
                 nll_reduction = "sum_per_sample",
                 ranking_loss_weight = 0.0,
                 ic_loss_weight = 0.0,
                 rank_temperature = 0.01,
                 rank_label_eps = 1e-8,
                 ic_eps = 1e-8):
        super().__init__()
        self.cs_scale = cs_scale
        self.nll_loss_weight = nll_loss_weight
        self.ret_loss_weight = ret_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.nll_reduction = nll_reduction
        self.ranking_loss_weight = ranking_loss_weight
        self.ic_loss_weight = ic_loss_weight
        self.rank_temperature = rank_temperature
        self.rank_label_eps = rank_label_eps
        self.ic_eps = ic_eps

    def __str__(self):
        return (
            f"SingleVAELoss(cs_scale = {self.cs_scale}, nll_loss_weight={self.nll_loss_weight}, "
            f"ret_loss_weight={self.ret_loss_weight}, kl_loss_weight={self.kl_loss_weight}, "
            f"nll_reduction={self.nll_reduction}, ranking_loss_weight={self.ranking_loss_weight}, "
            f"ic_loss_weight={self.ic_loss_weight}, rank_temperature={self.rank_temperature})"
        )

    def _pairwise_ranking_loss(self, pred_label: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred_label = pred_label.reshape(pred_label.shape[0], -1)
        label = label.reshape(label.shape[0], -1)

        pred_diff = pred_label.unsqueeze(-1) - pred_label.unsqueeze(-2)
        label_diff = label.unsqueeze(-1) - label.unsqueeze(-2)
        label_sign = torch.sign(label_diff)

        num_assets = label.shape[-1]
        pair_mask = torch.triu(
            torch.ones(num_assets, num_assets, dtype=torch.bool, device=label.device),
            diagonal=1,
        )
        pair_mask = pair_mask.unsqueeze(0) & (label_diff.abs() > self.rank_label_eps)

        if not torch.any(pair_mask):
            return pred_label.new_zeros(())

        temperature = max(float(self.rank_temperature), self.ic_eps)
        pair_losses = F.softplus(-label_sign * pred_diff / temperature)
        return pair_losses[pair_mask].mean()

    def _ic_loss(self, pred_label: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred_label = pred_label.reshape(pred_label.shape[0], -1)
        label = label.reshape(label.shape[0], -1)

        pred_centered = pred_label - pred_label.mean(dim=1, keepdim=True)
        label_centered = label - label.mean(dim=1, keepdim=True)

        covariance = (pred_centered * label_centered).mean(dim=1)
        pred_std = torch.sqrt((pred_centered ** 2).mean(dim=1)).clamp_min(self.ic_eps)
        label_std = torch.sqrt((label_centered ** 2).mean(dim=1))
        valid = label_std > self.ic_eps

        if not torch.any(valid):
            return pred_label.new_zeros(())

        ic = covariance[valid] / (pred_std[valid] * label_std[valid].clamp_min(self.ic_eps))
        ic = torch.nan_to_num(ic, nan=0.0, posinf=0.0, neginf=0.0)
        return 1.0 - ic.mean()

    def forward(
        self,
        sample,
        target_sample,
        label,
        pred_label,
        posterior,
        prior,
        mask=None,
        if_mask=False,
    ):
        """
        :param sample: (N, L, D)
        :param target_sample: (N, L, D)
        :param mask: (N, L)
        :param if_mask: bool
        :return: loss dict
        """

        assert sample.shape == target_sample.shape

        rec_loss = (sample - target_sample) ** 2
        nll_loss = rec_loss

        if if_mask:
            mask = mask.repeat(1, 1, nll_loss.shape[-1])
            nll_loss = nll_loss * mask

        if self.nll_reduction == "mean_like_ret":
            if if_mask:
                denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)
                nll_loss = nll_loss.sum(dim=(-1, -2)) / denom
            else:
                nll_loss = nll_loss.mean(dim=(-1, -2))
            weighted_nll_loss = nll_loss
            if self.nll_loss_weight is not None:
                weighted_nll_loss = self.nll_loss_weight * nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        elif self.nll_reduction == "sum_per_sample":
            weighted_nll_loss = nll_loss
            if self.nll_loss_weight is not None:
                weighted_nll_loss = self.nll_loss_weight * nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        else:
            raise ValueError(f"Unsupported nll_reduction: {self.nll_reduction}")

        ret_loss = (label - pred_label) ** 2
        ret_loss = ret_loss.mean(dim=-1)

        weighted_ret_loss = ret_loss
        if self.ret_loss_weight is not None:
            weighted_ret_loss = self.ret_loss_weight * ret_loss
        weighted_ret_loss = torch.sum(weighted_ret_loss) / weighted_ret_loss.shape[0]

        kl_loss = posterior.kl(prior, dims=[1])
        weighted_kl_loss = kl_loss
        if self.kl_loss_weight is not None:
            weighted_kl_loss = self.kl_loss_weight * kl_loss
        weighted_kl_loss = torch.sum(weighted_kl_loss) / weighted_kl_loss.shape[0]

        ranking_loss = self._pairwise_ranking_loss(pred_label, label)
        weighted_ranking_loss = ranking_loss
        if self.ranking_loss_weight is not None:
            weighted_ranking_loss = self.ranking_loss_weight * ranking_loss

        ic_loss = self._ic_loss(pred_label, label)
        weighted_ic_loss = ic_loss
        if self.ic_loss_weight is not None:
            weighted_ic_loss = self.ic_loss_weight * ic_loss

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
            weighted_ret_loss=weighted_ret_loss,
            weighted_kl_loss=weighted_kl_loss,
            ranking_loss=ranking_loss,
            weighted_ranking_loss=weighted_ranking_loss,
            ic_loss=ic_loss,
            weighted_ic_loss=weighted_ic_loss,
        )

        return loss_dict
