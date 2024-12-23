from typing import override
from functools import partial
from torchdatapipe.core.pipelines import DefaultDatasetPipeline, MultiplePipelines
from torchdatapipe.core.cache.writers import BinaryWriter
from torchdatapipe.core.datasets import BinaryDataset, identity
from torchdatapipe.core.codecs.numpy import NumpyCodec
from torchdatapipe.collections.vision.codecs.cv2 import PNGCodec
from torchdatapipe.collections.vision.types import ImageScene
from motivcv.collections.segmentation.data.types import SegmentationAnnotation


# cv2 любит отрезать канал у изображений типа HxWx1 и надо это отменить
def fix_chanel_multilabel(arr):
    if len(arr.shape) == 2:
        arr = arr[..., None]
    return arr


def dict2item_fn(data):
    segmentation = data["segmentation"]
    weight = data.get("weight")
    annotation = SegmentationAnnotation(segmentation, weight)
    return ImageScene(id=data["id"], image=data["image"], annotation=annotation)


def item2dict_fn(item, fix_chanel_fn):
    segmentation = fix_chanel_fn(item.annotation.segmentation)
    data = dict(id=item.id, image=item.image, segmentation=segmentation)
    if item.annotation.weight is not None:
        data["weight"] = fix_chanel_fn(item.annotation.weight)
    if item.annotation.metadata is not None:
        data["metadata"] = item.annotation.metadata
    return data


class CachedSegmentationPipeline(DefaultDatasetPipeline, MultiplePipelines):
    def __init__(self, root, multilabel=False):
        super().__init__(root)
        self.portion = root
        self.multilabel = multilabel
        self.code = dict(image=PNGCodec(), segmentation=NumpyCodec(), weight=NumpyCodec())

    @override
    def get_writer(self, writer_root, **kwargs):
        fix_chanel_fn = fix_chanel_multilabel if self.multilabel else identity
        _item2dict_fn = partial(item2dict_fn, fix_chanel_fn=fix_chanel_fn)
        writer = BinaryWriter(writer_root, self.code, _item2dict_fn, fast_keys=["metadata"])
        return writer

    @override
    def get_dataset(self, writer_root, imgsz, **kwargs):
        return BinaryDataset(writer_root, self.code, dict2item_fn, self.transform_fn(imgsz))

    @override
    def transform_fn(self, imgsz):
        return identity
