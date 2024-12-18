import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def _annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def _one_cycle(start, end, pct):
    y1 = start
    y2 = end
    # lambda function for sinusoidal ramp from y1 to y2
    return ((1 - math.cos(pct * math.pi)) / 2) * (y2 - y1) + y1


def _warmup_one_cycle_lr(step, warmup_steps, max_steps, lrf, lri=0):
    if step < warmup_steps:
        return lri + (1 - lri) * (step + 1) / (warmup_steps + 1)
    else:
        pct = (step - warmup_steps) / (max_steps - warmup_steps)
        return _one_cycle(1, lrf, pct)


def WarmupOneCycle(optimizer, warmup_steps, max_steps, lrf, lri=0):
    # assert warmup_steps < max_steps
    # lri = list(lri)
    # if len(lri) == 1:
    #     lri = lri[0]
    #     lr_fn = partial(_warmup_one_cycle_lr, warmup_steps=warmup_steps, max_steps=max_steps, lrf=lrf, lri=lri)
    # else:
    lr_fn = [
        partial(
            _warmup_one_cycle_lr,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            lrf=lrf,
            lri=_lri,
        )
        for _lri in lri
    ]
    return LambdaLR(optimizer, lr_lambda=lr_fn)  # , verbose=True)
