import einops as eo
import lightning.pytorch as pl
import lightning.pytorch.loggers.wandb
import torch
import torchmetrics as tm
from torch import nn

from ..ema import EMA
from .plots import render_and_close


@render_and_close
def plot_tm_metric(metric: tm.Metric):
    fig, ax = metric.plot()
    return fig


class Plots(pl.Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, task: pl.LightningModule):
        wandb_logger: lightning.pytorch.loggers.WandbLogger | None = trainer.logger
        if wandb_logger is None:
            return

        plots = [
            (metric_name, plot_tm_metric(metric))
            for metric_name, metric in task.val_plot_metrics.items()
        ]
        wandb_logger.log_image(
            key="val_plots",
            images=[img for _, img in plots],
            step=trainer.global_step,
            caption=[caption.removeprefix("Multiclass") for (caption, _) in plots],
        )
        return super().on_validation_epoch_end(trainer, task)

    def on_test_epoch_end(self, trainer: pl.Trainer, task: pl.LightningModule):
        wandb_logger: lightning.pytorch.loggers.WandbLogger | None = trainer.logger
        if wandb_logger is None:
            return

        plots = [
            (metric_name, plot_tm_metric(metric))
            for metric_name, metric in task.test_plot_metrics.items()
        ]
        wandb_logger.log_image(
            key="test_plots",
            images=[img for _, img in plots],
            step=trainer.global_step,
            caption=[caption.removeprefix("Multiclass") for (caption, _) in plots],
        )
        return super().on_test_epoch_end(trainer, task)


class SequenceClassification(pl.LightningModule):
    def __init__(self, model, n_classes: int, use_ema: bool, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = nn.CrossEntropyLoss()

        metrics = {
            "top1": tm.Accuracy(task="multiclass", num_classes=n_classes),
            "auc": tm.AUROC(task="multiclass", num_classes=n_classes),
        }
        if n_classes > 3:
            metrics["top3"] = tm.Accuracy(
                task="multiclass", num_classes=n_classes, top_k=3
            )
        metrics = tm.MetricCollection(metrics)

        plot_metrics = tm.MetricCollection(
            {
                "ConfusionMatrix": tm.ConfusionMatrix(
                    task="multiclass", num_classes=n_classes
                ),
                "ROC": tm.ROC(task="multiclass", num_classes=n_classes),
                "PRCurve": tm.PrecisionRecallCurve(
                    task="multiclass", num_classes=n_classes
                ),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.val_plot_metrics = plot_metrics.clone(prefix="val/")
        self.test_plot_metrics = plot_metrics.clone(prefix="test/")

        self.ema_model = None
        if use_ema:
            self.ema_model = EMA(
                model,
                beta=0.9999,
                update_after_step=1000,
                update_every=1,
                include_online_model=False,
                use_foreach=True,
            )

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = eo.reduce(self.model(x), "batch time classes -> batch classes", "mean")
        loss = self.loss(preds, y)

        self.log("train/loss", loss.mean(), batch_size=x.shape[0], prog_bar=True)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is not None:
            self.ema_model.update()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, self.val_metrics, self.val_plot_metrics)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, self.test_metrics, self.test_plot_metrics)

    def eval_step(self, batch, batch_idx, metrics, plot_metrics):
        x, y = batch

        model = self.ema_model or self.model
        preds = eo.reduce(model(x), "batch time classes -> batch classes", "mean")

        self.log_dict(metrics(preds, y), batch_size=x.shape[0])
        plot_metrics.update(preds, y)

    def configure_optimizers(self):
        ssm_matrix_parameters = [
            parameter
            for parameter in self.model.parameters()
            if hasattr(parameter, "ssm_matrix_lr")
        ]
        model_parameters = [
            parameter
            for parameter in self.model.parameters()
            if not hasattr(parameter, "ssm_matrix_lr")
        ]

        match self.hparams["optimizer"]["name"]:
            case "adam":
                optimizer = torch.optim.Adam(
                    model_parameters, lr=self.hparams["optimizer"]["lr"]
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    model_parameters,
                    lr=self.hparams["optimizer"]["lr"],
                    weight_decay=self.hparams["optimizer"]["weight_decay"],
                )
            case _:
                raise NotImplementedError("Pick an implemented optimizer.")

        if len(ssm_matrix_parameters) > 0:
            optimizer.add_param_group(
                {
                    "params": ssm_matrix_parameters,
                    "lr": self.hparams["optimizer"]["ssm_matrix_lr"],
                }
            )

        return optimizer

    def configure_callbacks(self):
        return super().configure_callbacks() + [Plots()]
