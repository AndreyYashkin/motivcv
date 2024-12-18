import cv2
import numpy as np
from typing import override
from dataclasses import dataclass
from torchdatapipe.core.utils.color import DefaultIndex2Color
from torchdatapipe.core.types import Visualizable
from torchdatapipe.collections.vision.types import Resizable
from torchdatapipe.collections.vision.utils.general import add_segmentation


_id2color = DefaultIndex2Color()
_dtype = np.uint8
_ERROR = np.iinfo(_dtype).max


@dataclass
class SegmentationAnnotation(Visualizable, Resizable):
    segmentation: np.array  # shape = (h, w) или (h, w, c)
    weight: np.array = None  # shape = (h, w) или (h, w, c)
    metadata: dict = None

    @override
    def visualize(self, image, channel=None, **kwargs):
        segmentation = self.segmentation
        if channel is not None:
            segmentation = segmentation[..., channel]
        segm_color = _id2color[segmentation]
        image = add_segmentation(image, segm_color)
        return image

    @override
    def resize(self, new_size, old_size=None, **kwargs):
        self.segmentation = cv2.resize(
            self.segmentation, new_size[::-1], interpolation=cv2.INTER_NEAREST
        )
