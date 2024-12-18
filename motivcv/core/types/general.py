from enum import StrEnum
from collections import namedtuple

AfterBatchTransferTuple = namedtuple("AfterBatchTransferTuple", ["input", "annotation", "items"])


class ModuleOutputs(StrEnum):
    MODEL = "model"
    LOSS = "loss"
    PREDICT = "predict"
