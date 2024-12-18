import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def create_single_torch_optimizer(optimizer_name, parameters, hyper_parameters):
    optimizer = getattr(optim, optimizer_name)
    if optimizer:
        return optimizer(parameters, **hyper_parameters)
    else:
        assert False, f'Unknown optimizer: "{optimizer_name}"'


def create_torch_optimizers(optimizer_names, parameters_groups: list, hyper_parameters):
    if not isinstance(optimizer_names, list):
        optimizer_names = [optimizer_names]
    if not isinstance(hyper_parameters, list):
        hyper_parameters = [hyper_parameters]

    optimizers = max(len(optimizer_names), len(hyper_parameters))
    fall = False
    if len(optimizer_names) == 1:
        optimizer_names = optimizer_names * optimizers
    elif len(optimizer_names) != optimizers:
        fall = True
    if len(parameters_groups) == 1:
        parameters_groups = parameters_groups * optimizers
    elif len(parameters_groups) != optimizers:
        fall = True
    if len(hyper_parameters) == 1:
        hyper_parameters = hyper_parameters * optimizers
    elif len(hyper_parameters) != optimizers:
        fall = True
    if fall:
        assert False, "The number of optimizers to create is unclear"

    optimizers = list()
    for optimizer_name, parameters, hyper_parameters_ in zip(
        optimizer_names, parameters_groups, hyper_parameters
    ):
        optimizer = create_single_torch_optimizer(optimizer_name, parameters, hyper_parameters_)
        optimizers.append(optimizer)
    return optimizers


def create_single_torch_scheduler(scheduler_name, optimizer, hyper_parameters):
    scheduler = getattr(lr_scheduler, scheduler_name)
    if scheduler:
        print("hyper_parameters", hyper_parameters)
        return scheduler(optimizer, **hyper_parameters)
    else:
        assert False, f'Unknown scheduler: "{scheduler_name}"'


def create_torch_schedulers(scheduler_names, optimizers: list, hyper_parameters):
    if not isinstance(scheduler_names, list):
        scheduler_names = [scheduler_names]
    if not isinstance(hyper_parameters, list):
        hyper_parameters = [hyper_parameters]

    fall = False
    if len(scheduler_names) == 1:
        scheduler_names = scheduler_names * len(optimizers)
    if len(hyper_parameters) == 1:
        hyper_parameters = hyper_parameters * len(optimizers)
    elif len(hyper_parameters) != len(optimizers):
        fall = True
    if fall:
        assert False, "The number of schedulers to create is unclear"

    schedulers = list()
    for scheduler_name, optimizer, hyper_parameters_ in zip(
        scheduler_names, optimizers, hyper_parameters
    ):
        scheduler = create_single_torch_scheduler(scheduler_name, optimizer, hyper_parameters_)
        schedulers.append(scheduler)
    return schedulers


# N optimizers, N schedulers
def configure_optimizers(
    optimizer_names,
    parameters_groups: list,
    scheduler_names=None,
    o_hyper_parameters=dict(),
    s_hyper_parameters=dict(),
):
    optimizers = create_torch_optimizers(optimizer_names, parameters_groups, o_hyper_parameters)
    if scheduler_names:
        schedulers = create_torch_schedulers(scheduler_names, optimizers, s_hyper_parameters)
        return optimizers, schedulers
    else:
        return optimizers


def configure_optimizer(
    optimizer_name: str,
    parameters,
    scheduler_name=None,
    o_hyper_parameters=dict(),
    s_hyper_parameters=dict(),
):
    return configure_optimizers(
        optimizer_name,
        [parameters],
        scheduler_name,
        o_hyper_parameters,
        s_hyper_parameters,
    )
