import torch
from warnings import warn
# from lion_pytorch import Lion
from torch.optim import lr_scheduler

def warning_kwargs(*args, **kwargs) -> None:
    """Wanrs user every time there are args or kwargs that are ignored"""
    if args:
        warn(f"skipping args:\n{args}")
    if kwargs:
        warn(f"skipping kwargs:\n{kwargs}")

class NoScheduler:
    """Scheduler that does nothing"""
    def __init__(self, *args, **kwargs):
        warning_kwargs(*args, **kwargs)

    def step(self, *args, **kwargs):
        warning_kwargs(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        warning_kwargs(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        warning_kwargs(*args, **kwargs)

schedulers = {
    "NoScheduler": NoScheduler,
    "LambdaLR": lr_scheduler.LambdaLR,
    "MultiplicativeLR": lr_scheduler.MultiplicativeLR,
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "ConstantLR": lr_scheduler.ConstantLR,
    "LinearLR": lr_scheduler.LinearLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "PolynomialLR": lr_scheduler.PolynomialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "ChainedScheduler": lr_scheduler.ChainedScheduler,
    "SequentialLR": lr_scheduler.SequentialLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CyclicLR": lr_scheduler.CyclicLR,
    # "OneCycleLR": lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts
}

optimizers = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
    # "Lion": Lion,
}

def construct_probKLLoss(*args, reduction='batchmean', **kwargs):
    kl_loss = torch.nn.KLDivLoss(*args, reduction=reduction, **kwargs)
    def probKLLoss(pred, target, *args, **kwargs):
        return kl_loss(torch.log(pred + 1e-7), target)
    return probKLLoss

losses = {
    'BCELoss':torch.nn.BCELoss,
    'KLDivLoss': torch.nn.KLDivLoss,
    'probKLLoss': construct_probKLLoss
}
