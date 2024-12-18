from typing import override
import cv2
import numpy as np
from pytorch_lightning.callbacks import Callback

DEFAULT_MAX_BS = 16


def add_id(image, id_text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (25, 25)
    # fontScale
    fontScale = 0.7
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(image, id_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return image


class DataVisualizationBase(Callback):
    def __init__(self, key):
        self.key = key
        self.train_saved = False
        self.valid_saved = False

    @override
    def on_fit_start(self, trainer, pl_module):
        self.train_saved = False
        self.valid_saved = False

    @override
    def on_train_end(self, trainer, pl_module):
        tensorboard_logger = pl_module.logger.experiment
        tensorboard_logger.add_image(
            f"train_batch_{self.key}",
            self.train_images,
            pl_module.global_step,
            dataformats="NHWC",
        )
        tensorboard_logger.add_image(
            f"valid_batch_{self.key}",
            self.valid_images,
            pl_module.global_step,
            dataformats="NHWC",
        )

    @override
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.train_saved:
            return
        logger = pl_module.logger.experiment
        self.train_images = self.get_images(pl_module, batch)
        logger.add_image(f"train_batch_{self.key}", self.train_images, dataformats="NHWC")
        self.train_saved = True

    @override
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.valid_saved:
            return
        logger = pl_module.logger.experiment
        self.valid_images = self.get_images(pl_module, batch)
        logger.add_image(f"valid_batch_{self.key}", self.valid_images, dataformats="NHWC")
        self.valid_saved = True

    def get_images(self, pl_module, batch):
        raise NotImplementedError


class DataVisualization(DataVisualizationBase):
    def __init__(self, max_bs=DEFAULT_MAX_BS, grayscale=False):
        super().__init__("image")
        self.max_bs = max_bs
        self.grayscale = grayscale

    @override
    def get_images(self, pl_module, batch):
        _, _, items = batch

        if len(items) > self.max_bs:
            items = items[: self.max_bs]

        images = []
        for item in items:
            image = item.visualize(grayscale=self.grayscale)
            images.append(image)

        return np.stack(images)


class DataAnnotationVisualization(DataVisualizationBase):
    def __init__(self, max_bs=DEFAULT_MAX_BS, suffix="", transform_fn=None, extra_kwargs=dict()):
        super().__init__(f"annotation{suffix}")
        self.max_bs = max_bs
        self.transform_fn = transform_fn
        self.extra_kwargs = extra_kwargs

    @override
    def get_images(self, pl_module, batch):
        _, _, items = batch

        if len(items) > self.max_bs:
            items = items[: self.max_bs]

        images = []
        for item in items:
            if self.transform_fn is not None:
                item = self.transform_fn(item)
            image = item.visualize(annotation=True, **self.extra_kwargs)
            image = add_id(image, str(item.id))
            images.append(image)

        return np.stack(images)
