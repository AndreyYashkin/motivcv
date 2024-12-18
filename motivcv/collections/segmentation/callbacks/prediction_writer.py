import os
from typing import override
import cv2
import numpy as np
from pytorch_lightning.callbacks import Callback
from torchdatapipe.core.utils.color import DefaultIndex2Color
from torchdatapipe.collections.vision.utils.general import add_segmentation


id2color = DefaultIndex2Color()


class MulticlassPredictionWriter(Callback):
    def __init__(self, root, ext=".jpg"):
        self.root = root
        self.ext = ext

    @override
    def on_predict_start(self, trainer, pl_module):
        os.makedirs(self.root, exist_ok=True)
        self.ii = 0

    @override
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        _, _, scenes = batch
        bs = len(scenes)

        mask = outputs["predict"].detach().cpu().numpy()

        mask = id2color[mask]
        mask = np.flip(mask, axis=-1)  # RGB -> BGR

        for i in range(bs):
            name = str(scenes[i].id)
            name = name.replace("/", "___")
            image = scenes[i].image

            cv2.imwrite(os.path.join(self.root, f"{self.ii}_{name}_img{self.ext}"), image)
            cv2.imwrite(os.path.join(self.root, f"{self.ii}_{name}_segm.png"), mask[i])

            debug = add_segmentation(image, mask[i])
            debug = np.flip(debug, axis=-1)  # RGB -> BGR
            cv2.imwrite(os.path.join(self.root, f"{self.ii}_{name}_debug{self.ext}"), debug)

            self.ii += 1
