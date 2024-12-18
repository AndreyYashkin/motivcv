from typing import override
from torchdatapipe.core.pipelines import TransformPipeline
from motivcv.collections.segmentation.data.datasets import AlbumentatedDataset
from motivcv.core.data.pipelines import ReplaceBackgroundPipeline as ReplaceBackgroundPipelineB


class AlbumentatedPipeline(TransformPipeline):
    def __init__(self, pipeline, configs):
        super().__init__(pipeline)
        self.configs = list(configs)

    @override
    def setup(self, *args, **kwargs):
        self.pipeline.setup(*args, **kwargs)
        self.__dataset = AlbumentatedDataset.from_config(self.pipeline.dataset, self.configs)

    @property
    def dataset(self):
        return self.__dataset


class ReplaceBackgroundPipeline(ReplaceBackgroundPipelineB):
    def __init__(self, pipeline, back_pipeline, p, background_cls=0):
        self.background_cls = background_cls
        super().__init__(pipeline, back_pipeline, self.get_backround_mask_fn, p)

    @override
    def get_backround_mask_fn(self, item):
        return item.annotation.segmentation == self.background_cls
