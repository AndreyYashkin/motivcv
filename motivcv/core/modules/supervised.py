from abc import ABC, abstractmethod
from typing import override
import torch
import pytorch_lightning as pl
from motivcv.core.data.datamodule import AfterBatchTransferTuple
from motivcv.core.types import ModuleOutputs


class SupervisedAlgorithm(ABC):
    @abstractmethod
    def organize_tensors(self, output: object) -> object:
        pass

    @abstractmethod
    def prepare_tensors_for_loss(self, output: object, annotation: object) -> object:
        pass

    @abstractmethod
    def compute_losses(self, tensors: dict) -> dict:
        pass

    @abstractmethod
    def balance_loss(self, losses: dict) -> torch.Tensor:
        pass

    @abstractmethod
    def get_predicts(self, output: object) -> object:
        pass


class SupervisedModule(pl.LightningModule):
    def __init__(
        self,
        algorithm: SupervisedAlgorithm,
        valid_predict: bool = False,
        test_predict: bool = False,
    ):
        super().__init__()
        # self.model = model
        self.algorithm = algorithm
        self.__valid_predict = valid_predict
        self.__test_predict = test_predict
        # self._example_input_array = "TODO"

    # @override
    # def forward(self, image: torch.Tensor) -> object:
    #     return self.model(image)

    @override
    def fit_step(self, batch: AfterBatchTransferTuple, stage: str, log_on_epoch: bool) -> dict:
        image, labels, scenes = batch
        batch_size = len(scenes)

        outs = self(image)
        outs = self.algorithm.organize_tensors(outs)
        tensors = self.algorithm.prepare_tensors_for_loss(outs, labels)
        losses = self.algorithm.compute_losses(tensors)
        balanced_loss = self.algorithm.balance_loss(losses)

        self.log(f"loss/{stage}", balanced_loss, on_epoch=log_on_epoch, batch_size=batch_size)

        if len(losses) > 1:
            for name, loss in losses.items():
                self.log(f"{name}_loss/{stage}", loss, on_epoch=log_on_epoch, batch_size=batch_size)

        return {
            ModuleOutputs.MODEL: outs,
            ModuleOutputs.LOSS: balanced_loss,
        }

    @override
    def training_step(self, batch: AfterBatchTransferTuple, batch_idx):
        return self.fit_step(batch, "train", log_on_epoch=False)

    @override
    def valid_test_step(self, batch, stage, do_predict):
        result = self.fit_step(batch, stage, log_on_epoch=True)
        if do_predict:
            outs = result[ModuleOutputs.MODEL]
            predict = self.algorithm.get_predicts(outs)
            result[ModuleOutputs.PREDICT] = predict
        return result

    @override
    def validation_step(self, batch: AfterBatchTransferTuple, batch_idx):
        return self.valid_test_step(batch, "val", self.__valid_predict)

    @override
    def test_step(self, batch: AfterBatchTransferTuple, batch_idx):
        return self.valid_test_step(batch, "test", self.__test_predict)

    @override
    def predict_step(self, batch: AfterBatchTransferTuple, batch_idx):
        image, _, scenes = batch
        # TODO
        image = image.to(torch.float)
        outs = self(image)
        outs = self.algorithm.organize_tensors(outs)
        predict = self.algorithm.get_predicts(outs)
        return {
            ModuleOutputs.MODEL: outs,
            ModuleOutputs.PREDICT: predict,
        }

    # @override
    # def configure_optimizers(self):
    #
