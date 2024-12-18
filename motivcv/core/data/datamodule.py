from typing import override

# from abc import ABC, abstractmethod
import numpy as np
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torchdatapipe.core.pipelines import PipelineCacheDescription
from motivcv.core.types import AfterBatchTransferTuple


class DataModule(L.LightningDataModule):  # , ABC):
    def __init__(
        self,
        data_prefix,
        cache_dir,
        imgsz,
        train,
        valid,
        test,
        predict,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_prefix = data_prefix
        self.cache_dir = cache_dir
        self.imgsz = imgsz
        self.train = self.create_pipeline(train)
        self.valid = self.create_pipeline(valid)
        self.test = self.create_pipeline(test)
        self.predict = self.create_pipeline(predict)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.prepare_data_per_node = False

    # @abstractmethod
    def create_pipeline(self, cfg):
        raise NotImplementedError

    # @abstractmethod
    def collate_fn(self, scenes):
        raise NotImplementedError

    # @abstractmethod
    def predict_collate_fn(self, scenes):
        raise NotImplementedError

    # @abstractmethod
    def on_after_batch_transfer(self, batch, dataloader_idx) -> AfterBatchTransferTuple:
        raise NotImplementedError

    @override
    def prepare_data(self) -> None:
        cache_desc = []
        if self.train is not None:
            cache_desc += self.train.get_cache_desc(
                self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz
            )
        if self.valid is not None:
            cache_desc += self.valid.get_cache_desc(
                self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz
            )
        if self.test is not None:
            cache_desc += self.test.get_cache_desc(
                self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz
            )

        conflicts, err = PipelineCacheDescription.check_conflicts(cache_desc)
        assert not conflicts, err

    @override
    def setup(self, stage: str):
        if stage == "predict":
            self.predict.setup(self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz)
            return

        if stage == "fit" and self.train is not None:
            self.train.setup(self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz)
        if stage in ["fit", "valid"] and self.valid is not None:
            self.valid.setup(self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz)
        if stage in ["fit", "test"] and self.test is not None:
            self.test.setup(self.data_prefix, cache_dir=self.cache_dir, imgsz=self.imgsz)

    @override
    def train_dataloader(self, as_valid=False):
        if self.train is None:
            return []
        dataset = self.train.dataset
        # TODO seed
        sampler = self.train.get_sampler(shuffle=not as_valid)
        rng = np.random.default_rng()
        sampler.set_rng(rng)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=not as_valid,
        )

    @override
    def val_dataloader(self):
        if self.valid is None:
            return []
        dataset = self.valid.dataset
        sampler = self.valid.get_sampler(shuffle=False)
        # rng = np.random.default_rng()
        # sampler.set_rng(rng)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )

    @override
    def test_dataloader(self):
        if self.test is None:
            return []
        dataset = self.test.dataset
        sampler = self.test.get_sampler(shuffle=False)
        # rng = np.random.default_rng()
        # sampler.set_rng(rng)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )

    @override
    def predict_dataloader(self):
        if self.predict is None:
            return []
        dataset = self.predict.dataset
        sampler = self.predict.get_sampler(shuffle=False)
        # rng = np.random.default_rng()
        # sampler.set_rng(rng)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.predict_collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )
