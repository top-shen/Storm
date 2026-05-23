import numpy as np

from storm.registry import DOWNSTREAM
from storm.metrics import ARR, SR, SOR, CR, MDD, VOL, DD

@DOWNSTREAM.register_module(force=True)
class TopkDropoutStrategy():
    def __init__(self,
                 *args,
                 topk: int = 5,
                 dropout: int = 3,
                 transaction_cost_ratio: float = 1e-4,
                 init_cash: float = 1e6,
                 **kwargs):
        self.topk = topk
        self.dropout = dropout
        self.transaction_cost_ratio = transaction_cost_ratio
        self.init_cash = init_cash

    def __call__(self,
                 *args,
                 pred_labels: np.ndarray,
                 true_labels: np.ndarray,
                 **kwargs):
        """
        :param args:
        :param pred_labels: (n_days, n_assets)
        :param true_labels: (n_days, n_assets)
        :param kwargs:
        :return:
        """

        N, S = pred_labels.shape
        assert N == true_labels.shape[0] and S == true_labels.shape[1], "Prediction and true labels shape mismatch."
        k = min(int(self.topk), S)
        dropout = min(int(self.dropout), k)
        min_hold = max(k - dropout, 0)

        value = self.init_cash
        rets = []
        culmulative_rets = []

        # Initialize the top-k list with the first day's predictions
        topk_list = np.argsort(pred_labels[0])[-k:]

        for i in range(N):
            pred_label = pred_labels[i].flatten()
            true_label = true_labels[i].flatten()
            turnover = 0.0

            if i > 0:
                pre_topk_list = topk_list.copy()

                # Find intersection of previous and current top-k
                same_assets_set = set(pre_topk_list) & set(np.argsort(pred_label)[-k:])

                # Determine hold assets
                hold_assets = list(same_assets_set)

                # Ensure at least topk - dropout assets are retained
                if len(hold_assets) < min_hold:
                    # Sort remaining assets by predicted score
                    remaining_assets = list(set(pre_topk_list) - same_assets_set)
                    sorted_remaining = sorted(remaining_assets, key=lambda x: pred_label[x])

                    # Retain additional assets to meet topk - dropout requirement
                    hold_assets.extend(sorted_remaining[-(min_hold - len(hold_assets)):])

                # Update the top-k list
                non_hold_assets = list(set(range(S)) - set(hold_assets))
                sorted_non_hold = sorted(non_hold_assets, key=lambda x: pred_label[x])

                # Select the top-k performing assets and ensure they are sorted in ascending order
                topk_candidates = hold_assets + sorted_non_hold[-(k - len(hold_assets)):]
                topk_list = sorted(topk_candidates, key=lambda x: pred_label[x])[-k:]

                turnover = 2.0 * (k - len(set(pre_topk_list) & set(topk_list))) / k

            # Calculate returns and update value
            gross_ret = np.mean(true_label[topk_list])
            ret = gross_ret - self.transaction_cost_ratio * turnover
            value += value * ret
            rets.append(ret)

            culmulative_ret = (value - self.init_cash) / self.init_cash
            culmulative_rets.append(culmulative_ret)

        rets = np.array(rets)
        arr = ARR(rets)
        sr = SR(rets)
        dd = DD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)
        CW = culmulative_rets[-1]

        return_info = {
            "CW": CW,
            "ARR%": arr * 100,
            "SR": sr,
            "CR": cr,
            "SOR": sor,
            "DD": dd,
            "MDD%": mdd * 100,
            "VOL": vol
        }

        return return_info


if __name__ == '__main__':
    pred_labels = np.random.rand(4, 10)
    true_labels = np.random.rand(4, 10)

    strategy = TopkDropoutStrategy()
    strategy(pred_labels=pred_labels, true_labels=true_labels)
