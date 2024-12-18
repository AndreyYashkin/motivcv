import torch
import numpy as np
from typing import override

from torchdatapipe.collections.vision.utils.general import cv2_image_collate_fn
from motivcv.core.data.datamodule import DataModule


class SegmentationDataModule(DataModule):
    @property
    def pixel_weights(self) -> bool:
        raise NotImplementedError

    @override
    def collate_fn(self, scenes):
        images = [item.image for item in scenes]
        segmentations = [item.annotation.segmentation for item in scenes]

        images = cv2_image_collate_fn(images)
        images = torch.tensor(images, dtype=torch.float)
        segmentations = np.stack(segmentations)
        segmentation = torch.tensor(segmentations, dtype=torch.long)
        if len(segmentation.shape) == 4:
            segmentation = torch.movedim(segmentation, -1, 1)

        b = dict(images=images, segmentation=segmentation, scenes=scenes)
        if self.pixel_weights:
            weights = [
                (
                    item.annotation.weight
                    if item.annotation.weight is not None
                    else np.ones_like(item.annotation.segmentation, dtype=float)
                )
                for item in scenes
            ]
            weights = np.stack(weights)
            weights = torch.tensor(weights, dtype=torch.float)
            if len(weights.shape) == 4:
                weights = torch.movedim(weights, -1, 1)
            b["weights"] = weights
        return b

    @override
    def predict_collate_fn(self, scenes):
        images = [item.image for item in scenes]
        image = cv2_image_collate_fn(images)
        image = torch.tensor(image)
        return dict(images=image, scenes=scenes)

    @override
    def on_after_batch_transfer(self, batch, dataloader_idx):
        images = batch["images"]
        scenes = batch["scenes"]
        if "segmentation" in batch:
            annotation = dict(segmentation=batch["segmentation"])
            if self.pixel_weights:
                annotation["weights"] = batch["weights"]
        else:
            annotation = None
        return images, annotation, scenes
