from typing import override
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

try:
    from thop import profile, clever_format

    _thop_available = True
except ImportError:
    _thop_available = False


# Не очень полезная штука, т.к. не умеет работать с кастомные
class Thtop(Callback):
    custom_ops = {}

    def __init__(self, imgsz: tuple[int, int]):
        assert _thop_available
        self.imgsz = imgsz

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        image = torch.zeros(1, 3, *self.imgsz).to(pl_module.device)
        model = pl_module.model
        macs, params = profile(
            model, inputs=(image,), custom_ops=self.custom_ops, report_missing=True
        )
        macs, params = clever_format([macs, params], "%.3f")
        log = {
            "MACs": macs,
            "Params": params,
        }
        print("Thtop:", log)
        # pl_module.logger.log_hyperparams(log)
