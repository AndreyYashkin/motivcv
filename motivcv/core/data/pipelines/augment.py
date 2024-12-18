from typing import override
from torchdatapipe.core.pipelines import BaseMultiIndexDataPipeline
from torchdatapipe.core.samplers import StackDatasetSampler
from torchdatapipe.collections.vision.datasets import ReplacedBackgroundDataset


class ReplaceBackgroundPipeline(BaseMultiIndexDataPipeline):
    def __init__(self, pipeline, back_pipeline, get_background_mask_fn, p):
        super().__init__([pipeline, back_pipeline])
        self.pipeline = pipeline
        self.back_pipeline = back_pipeline
        self.get_background_mask_fn = get_background_mask_fn
        self.p = p

    @override
    def setup(self, *args, **kwargs):
        self.pipeline.setup(*args, **kwargs)
        self.back_pipeline.setup(*args, **kwargs)

        # get_backround_mask_fn = (
        #     lambda item: item.annotation.segmentation == self.back_id
        # )

        self.__dataset = ReplacedBackgroundDataset(
            self.pipeline.dataset,
            self.back_pipeline.dataset,
            self.get_background_mask_fn,
            self.p,
        )

    @override
    @property
    def dataset(self):
        return self.__dataset

    @override
    def get_sampler(self, shuffle=True):
        sampler = self.pipeline.get_sampler(shuffle)
        back = self.back_pipeline.get_sampler(shuffle)
        return StackDatasetSampler([sampler, back])

    @override
    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        pass
