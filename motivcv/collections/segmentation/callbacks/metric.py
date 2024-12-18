# import torch
from typing import override, Any
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
import pytorch_lightning as pl

# from motivcv.collections.segmentation.data
from motivcv.core.types import ModuleOutputs


class MulticlassPixelMetrics(Callback):
    def __init__(self, class_names, std_dil=1.0):
        super().__init__()
        self.class_names = class_names
        self.std_dil = std_dil
        num_classes = len(class_names)

        precision = MulticlassPrecision(num_classes=num_classes, average="none")
        recall = MulticlassRecall(num_classes=num_classes, average="none")
        f1_score = MulticlassF1Score(num_classes=num_classes, average="none")
        self.true_confmat = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")
        self.pred_confmat = MulticlassConfusionMatrix(num_classes=num_classes, normalize="pred")
        self.full_confmat = MulticlassConfusionMatrix(num_classes=num_classes, normalize="none")

        self.metric_collection = MetricCollection([precision, recall, f1_score])

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device
        self.metric_collection.to(device)
        self.true_confmat.to(device)
        self.pred_confmat.to(device)
        self.full_confmat.to(device)

    @override
    def on_batch_end(self, outputs, batch):
        predict = outputs[ModuleOutputs.PREDICT]
        segmentation = batch[1]["segmentation"]  # FIXME !!!!!!!!!!!!
        self.metric_collection.update(predict, segmentation)
        self.true_confmat.update(predict, segmentation)
        self.pred_confmat.update(predict, segmentation)
        self.full_confmat.update(predict, segmentation)

    @override
    def on_epoch_end(self, pl_module, log_name):
        metrics = self.metric_collection.compute()

        for metric_name, values in metrics.items():
            pl_module.log(f"{log_name}_{metric_name}", values.mean())
            metric_d = {
                f"{log_name}_{metric_name}/{name}": val
                for name, val in zip(self.class_names, values)
            }

            pl_module.log_dict(metric_d)

        self.metric_collection.reset()
        logger = pl_module.logger.experiment
        global_step = pl_module.global_step

        fig, ax_ = self.true_confmat.plot(labels=self.class_names)
        self.true_confmat.reset()
        logger.add_figure(f"{log_name}_true_confmat", fig, global_step)

        fig, ax_ = self.pred_confmat.plot(labels=self.class_names)
        self.pred_confmat.reset()
        logger.add_figure(f"{log_name}_pred_confmat", fig, global_step)

        val = self.full_confmat.compute()
        val = val / self.std_dil
        fig, ax_ = self.full_confmat.plot(val, labels=self.class_names)
        self.full_confmat.reset()
        logger.add_figure(f"{log_name}_full_confmat", fig, global_step)

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_end(outputs, batch)

    @override
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.on_epoch_end(pl_module, "valid")

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_end(outputs, batch)

    @override
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_epoch_end(pl_module, "test")
