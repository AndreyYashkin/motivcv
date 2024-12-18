import cv2
import numpy as np
from typing import override
from torchdatapipe.core.cache.preprocessors import Preprocessor, Sequential, Identity


class AddBorderPixelWeights(Preprocessor):
    def __init__(self, weights: dict, mask_classes, is_in=True, policy=None):
        self.weights = dict(weights)
        self.mask_classes = list(mask_classes)
        self.is_in = is_in
        assert policy in [None, "max"]
        self.policy = policy

    @override
    def start_caching(self):
        pass

    @override
    def __call__(self, item):
        a = item.annotation
        weight = a.weight
        assert self.policy or a.weight is None
        if weight is None:
            weight = np.ones(a.segmentation.shape, dtype=float)

        mask = np.isin(a.segmentation, self.mask_classes).astype(np.uint8)
        if not self.is_in:
            mask = 1 - mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for r in sorted(self.weights.keys(), reverse=True):
            w = self.weights[r]
            mask = cv2.drawContours(np.zeros_like(a.segmentation), contours, -1, 255, r) > 0
            mask[mask] = weight[mask] < w
            weight[mask] = w

        item.annotation.weight = weight
        return item

    @override
    def finish_caching(self):
        pass

    @override
    @property
    def version(self):
        return "0.1.1"

    @override
    @property
    def params(self):
        return dict(
            weights=self.weights,
            mask_classes=self.mask_classes,
            is_in=self.is_in,
            policy=self.policy,
        )


def AddBorderPixelWeightsSequence(configs):
    if configs is None or not len(configs):
        return Identity()
    return Sequential([AddBorderPixelWeights(**cfg) for cfg in configs])
