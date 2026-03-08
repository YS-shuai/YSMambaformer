
from torch import nn


class MaskedLoss(nn.Module):
    def __init__(self, context_length=None, target_length=None, pred_mask_value=None, **kwargs):
        super(MaskedLoss, self).__init__()
        self.context_length = context_length
        self.target_length = target_length
        self.pred_mask_value = pred_mask_value

    def forward(self, preds, batch, aux, current_step=None):
        mask = ((batch["dynamic_mask"][0][:, self.context_length: self.context_length+self.target_length, ...,] < 1.0)
                .bool().type_as(preds))
        targets = batch["dynamic"][0][:, self.context_length: self.context_length+self.target_length, ...,]

        sum_squared_error = (((preds - targets) * mask) ** 2).sum(1)  # b c h w
        mse = sum_squared_error / (mask.sum(1) + 1e-8)  # b c h w
        if self.pred_mask_value:  # what is that?
            pred_mask = (
                (preds != self.pred_mask_value).bool().type_as(preds).max(1)[0]
            )
            mse = (mse * pred_mask).sum() / (pred_mask.sum() + 1e-8)

        logs = {"loss": mse}

        return mse, logs


class MaskedL2NDVILoss(nn.Module):
    def __init__(
        self,
        lc_min=None,
        lc_max=None,
        context_length=None,
        target_length=None,
        ndvi_pred_idx=0,
        ndvi_targ_idx=0,
        pred_mask_value=None,
        scale_by_std=False,
        weight_by_std=False,
        decouple_loss_term=None,
        decouple_loss_weight=1,
        posterior_loss_step=None,
        posterior_loss_term=None,
        posterior_loss_step1=None,
        posterior_loss_step2=None,
        posterior_loss_weight=1,
        mask_hq_only=False,
        **kwargs,
    ):
        super().__init__()

        self.lc_min = (
            lc_min if lc_min else None
        )  # landcover boudaries of vegetation (to select only pixel with vegetation)
        self.lc_max = lc_max if lc_max else None
        self.use_lc = lc_min & lc_max
        if not self.use_lc:
            print(
                f"WARNING. The boundaries of the landcover map are not definite. Loss calculated on all pixels including non-vegetation pixels."
            )
        self.context_length = context_length
        self.target_length = target_length
        self.ndvi_pred_idx = ndvi_pred_idx  # index of the NDVI band
        self.ndvi_targ_idx = ndvi_targ_idx  # index of the NDVI band
        self.pred_mask_value = pred_mask_value
        self.scale_by_std = scale_by_std
        self.weight_by_std = weight_by_std
        if self.scale_by_std:
            print(
                f"Using Masked L2/Std NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )
        else:
            print(
                f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )

        self.decouple_loss_term = decouple_loss_term
        self.decouple_loss_weight = decouple_loss_weight
        self.posterior_loss_term = posterior_loss_term
        self.posterior_loss_weight = posterior_loss_weight
        self.posterior_loss_step1 = posterior_loss_step1
        self.posterior_loss_step2 = posterior_loss_step2
        self.posterior_loss_step = posterior_loss_step
        self.mask_hq_only = mask_hq_only

    def forward(self, preds, batch, aux, current_step=None):
        # Mask
        # Cloud mask
        s2_mask = (
            (
                batch["dynamic_mask"][0][
                    :,
                    self.context_length : self.context_length + self.target_length,
                    ...,
                ]
                < 1.0
            )
            .bool()
            .type_as(preds)
        )  # b t c h w

        # Landcover mask
        lc = batch["landcover"]
        lc_mask = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(
            preds
        )  # b c h w
        ndvi_targ = batch["dynamic"][0][
            :,
            self.context_length : self.context_length + self.target_length,
            self.ndvi_targ_idx,
            ...,
        ].unsqueeze(
            2
        )  # b t c h w

        ndvi_pred = preds[:, -ndvi_targ.shape[1]:, self.ndvi_pred_idx, ...].unsqueeze(
            2
        )  # b t c h w

        sum_squared_error = (((ndvi_targ - ndvi_pred) * s2_mask) ** 2).sum(1)  # b c h w
        mse = sum_squared_error / (s2_mask.sum(1) + 1e-8)  # b c h w

        if self.scale_by_std:
            mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
                s2_mask.sum(1).unsqueeze(1) + 1e-8
            )  # b t c h w
            sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
                1
            )  # b c h w
            mse = sum_squared_error / sum_squared_deviation.clip(
                min=0.01
            )  # mse b c h w
        elif self.weight_by_std:
            mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
                s2_mask.sum(1).unsqueeze(1) + 1e-8
            )  # b t c h w
            sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
                1
            )  # b c h w
            mse = sum_squared_error * (
                ((sum_squared_deviation / (s2_mask.sum(1) + 1e-8)) ** 0.5) / 0.1
            ).clip(
                min=0.01, max=100.0
            )  # b c h w

        if self.pred_mask_value:  # what is that?
            pred_mask = (
                (ndvi_pred != self.pred_mask_value).bool().type_as(preds).max(1)[0]
            )
            mse_lc = (mse * lc_mask * pred_mask).sum() / (
                (lc_mask * pred_mask).sum() + 1e-8
            )
        elif self.use_lc:
            mse_lc = (mse * lc_mask).sum() / (lc_mask.sum() + 1e-8)
        else:
            mse_lc = mse.mean()

        logs = {"loss": mse_lc}

        if self.decouple_loss_term:
            extra_loss = aux[self.decouple_loss_term]
            logs["mse_lc"] = mse_lc
            logs[self.decouple_loss_term] = extra_loss
            mse_lc += self.decouple_loss_weight * extra_loss
            logs["loss"] = mse_lc

        if self.posterior_loss_term:
            extra_loss = aux[self.posterior_loss_term]
            logs["mse_lc"] = mse_lc
            logs[self.posterior_loss_term] = extra_loss
            pl_weight = 1
            if self.posterior_loss_step:
                if current_step < self.posterior_loss_step1:
                    pl_weight = 5
                elif current_step < self.posterior_loss_step2:
                    pl_weight = 5 - (4 / (self.posterior_loss_step2 - self.posterior_loss_step1)) * (
                                current_step - self.posterior_loss_step1)
                else:
                    pl_weight = 1

            mse_lc += self.posterior_loss_weight * extra_loss * pl_weight
            logs["loss"] = mse_lc

        return mse_lc, logs


class MSELoss(nn.Module):
    def __init__(
        self,
        context_length=None,
        target_length=None,
        **kwargs,
    ):
        super().__init__()
        self.context_length = context_length
        self.target_length = target_length
        self.MSE_criterion = nn.MSELoss()

    def forward(self, preds, batch, aux=None, current_step=None):
        targ = batch["dynamic"][:, self.context_length: self.context_length + self.target_length, ...,]
        pred = preds[:, -targ.shape[1]:, ...]
        mse = self.MSE_criterion(targ, pred)
        logs = {"loss": mse}
        return mse, logs


def setup_loss(args):
    if args["name"] == "MaskedL2NDVILoss":
        return MaskedL2NDVILoss(**args)
    elif args["name"] == "MSELoss":
        return MSELoss(**args)
    elif args["name"] == "masked":
        return MaskedLoss(**args)
