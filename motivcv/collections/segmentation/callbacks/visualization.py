from typing import override
import cv2
import numpy as np
from torchdatapipe.collections.vision.utils.general import add_segmentation
from motivcv.core.callbacks import DataVisualizationBase, DEFAULT_MAX_BS


class LossWeightVisualization(DataVisualizationBase):
    def __init__(self, max_bs=DEFAULT_MAX_BS, max_value=20):
        super().__init__("Loss_weight")
        self.max_bs = max_bs
        self.max_value = max_value

    @override
    def get_images(self, pl_module, batch):
        items = batch["scenes"]
        label = batch["label"]
        weights = batch.get("weights")

        if len(items) > self.max_bs:
            items = items[: self.max_bs]
            label = label[: self.max_bs]
            if weights is not None:
                weights = weights[: self.max_bs]

        weights = pl_module.build_loss_weight(label, weights).cpu().numpy()
        max_value = weights.max() if self.max_value is None else self.max_value
        weights = (np.clip(weights, 0, max_value) / max_value * 255).astype(np.uint8)

        images = []
        for item, w in zip(items, weights):
            image = item.visualizate(grayscale=True)
            w = cv2.applyColorMap(w, cv2.COLORMAP_PLASMA)
            w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
            image = add_segmentation(image, w)
            images.append(image)

        return np.stack(images)
