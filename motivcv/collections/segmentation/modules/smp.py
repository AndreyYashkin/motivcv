from typing import override
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from motivcv.core.models import ToGrayscale, Normalize
from motivcv.core.modules.supervised import SupervisedAlgorithm, SupervisedModule
from motivcv.legacy.optim.configure_torch_optimizers import create_single_torch_optimizer
from motivcv.legacy.optim.lr_scheduler import WarmupOneCycle


class MulticlassSegmentationAlgorithm(SupervisedAlgorithm, nn.Module):
    # focal_gamma = 0 - это обычная CE
    def __init__(
        self,
        # focal loss
        focal=1.0,  # баланс для focal_loss
        focal_cls_weight=None,
        focal_alpha=None,
        focal_gamma=0,
        # dice loss
        dice=None,
        dice_log_loss=False,
        dice_smooth=0.0,
    ):
        super().__init__()

        self.use_focal = focal is not None
        if self.use_focal:
            self.focal_balance = focal
            self.focal_loss = FocalLoss(
                mode="multiclass", alpha=focal_alpha, gamma=focal_gamma, reduction="none"
            )
            if focal_cls_weight is not None:
                self.register_buffer("focal_cls_weight", torch.tensor(focal_cls_weight))
            else:
                self.focal_cls_weight = None

        self.use_dice = dice is not None
        if self.use_dice:
            self.dice_balance = dice
            self.dice_loss = DiceLoss("multiclass", log_loss=dice_log_loss, smooth=dice_smooth)

    @override
    def organize_tensors(self, output) -> torch.Tensor:
        return output

    @override
    def prepare_tensors_for_loss(self, output, annotation) -> dict:
        segmentation = annotation["segmentation"]
        pixel_weights = annotation.get("weights")
        tensors = dict(output=output, segmentation=segmentation)
        if self.use_focal:
            if self.focal_cls_weight is None and pixel_weights is None:
                return dict(output=output, segmentation=segmentation)
            if self.focal_cls_weight is not None:
                weights = self.focal_cls_weight[segmentation]
            else:
                weights = torch.ones_like(segmentation, dtype=torch.float)
            if pixel_weights is not None:
                weights = torch.maximum(weights, pixel_weights)
            tensors["weights"] = weights
        return tensors

    @override
    def compute_losses(self, tensors: dict) -> dict:
        losses = dict()
        if self.use_focal:
            pixel_loss = self.focal_loss(tensors["output"], tensors["segmentation"])
            if "weights" in tensors:
                pixel_loss *= tensors["weights"]
            losses["focal"] = pixel_loss.mean()
        if self.use_dice:
            losses["dice"] = self.dice_loss(tensors["output"], tensors["segmentation"])
        return losses

    @override
    def balance_loss(self, losses: dict) -> torch.Tensor:
        _losses = []
        if self.use_focal:
            _losses.append(self.focal_balance * losses["focal"])
        if self.use_dice:
            _losses.append(self.dice_balance * losses["dice"])
        assert len(_losses)
        return torch.stack(_losses).sum()

    @override
    def get_predicts(self, output: torch.Tensor) -> object:
        return torch.argmax(output, dim=1)


class MulticlassSegmentationModule(SupervisedModule):
    def __init__(self, cfg):
        self.cfg = cfg

        self.class_names = cfg["model"]["class_names"]
        num_classes = len(self.class_names)
        modules = []

        if cfg["model"]["normalize"]:
            modules.append(Normalize(**cfg["model"]["normalize"]))

        if cfg["model"]["grayscale"]:
            modules.append(ToGrayscale(keep_channels=False))

        in_channels = 1 if cfg["model"]["grayscale"] else 3
        architecture = cfg["model"]["architecture"]
        assert "in_channels" not in architecture
        assert "classes" not in architecture

        model = smp.create_model(in_channels=in_channels, classes=num_classes, **architecture)
        model.encoder.set_swish(memory_efficient=False)
        modules.append(model)

        loss = cfg["model"]["loss"]
        algorithm = MulticlassSegmentationAlgorithm(**loss)

        super().__init__(algorithm, True, True)
        self.model = nn.Sequential(*modules) if len(modules) > 1 else model

    @override
    def forward(self, image: torch.Tensor) -> object:
        return self.model(image)

    @override
    def configure_optimizers(self):
        stepping_batches = self.trainer.estimated_stepping_batches
        assert stepping_batches > 0

        opt = self.cfg["optimization"]

        optimizer_name = opt["optimizer"]["name"]
        optimizer_hparams = opt["optimizer"]["hparams"]

        # scheduler_name = self.hparams.cfg.model.optim.scheduler.name
        max_steps_param = opt["scheduler"]["max_steps_param"]
        scheduler_hparams = opt["scheduler"]["hparams"]

        optimizer = create_single_torch_optimizer(
            optimizer_name, self.model.parameters(), optimizer_hparams
        )

        scheduler_hparams = dict(scheduler_hparams)
        scheduler_hparams[max_steps_param] = stepping_batches
        scheduler = WarmupOneCycle(optimizer, **scheduler_hparams)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
