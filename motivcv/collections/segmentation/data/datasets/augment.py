import albumentations as A
from torchdatapipe.core.datasets import TransformedDataset


class AlbumentatedDataset(TransformedDataset):
    def __init__(self, dataset, transform):
        self.transform = transform
        super().__init__(dataset, self.transform_fn)

    @staticmethod
    def from_config(dataset, configs, data_format="yaml"):
        transforms = []
        for config in configs:
            transform = A.load(config, data_format=data_format)
            transforms.append(transform)
        transform = A.Compose(transforms, additional_targets={"weight": "mask"})

        return AlbumentatedDataset(dataset, transform)

    def transform_fn(self, item):
        image = item.image

        albu_item = dict(image=image)
        a = item.annotation
        if a is not None:
            mask = a.segmentation
            albu_item["mask"] = mask
            if a.weight is not None:
                albu_item["weight"] = a.weight
        albu_item = self.transform(**albu_item)

        item.image = albu_item["image"]
        if "mask" in albu_item:
            a.segmentation = albu_item["mask"]
        if "weight" in albu_item:
            a.weight = albu_item["weight"]

        return item
