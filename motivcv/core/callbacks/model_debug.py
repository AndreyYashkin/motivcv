from typing import override
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import grad_norm


class ParamCallback(Callback):
    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        param_l = []
        for name, param in pl_module.named_parameters():
            with torch.no_grad():
                param_l.append(param.flatten())

        with torch.no_grad():
            params = torch.cat(param_l)

        std, mean = torch.std_mean(params)
        self.log("param_std", std)
        self.log("param_mean", mean)


class GradValuesCallback(Callback):
    @override
    def on_after_backward(self, trainer, pl_module):
        grad_l = []
        for name, param in pl_module.named_parameters():
            if isinstance(param.grad, type(None)):
                continue
            grad_l.append(param.grad.flatten())

        grads = torch.cat(grad_l)

        std, mean = torch.std_mean(grads)
        self.log("grad_std", std)
        self.log("grad_mean", mean)


class GradNormCallback(Callback):
    @override
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(pl_module.model, norm_type=2)
        norm_total = norms["grad_2.0_norm_total"]
        self.log("grad_2.0_norm_total", norm_total)
