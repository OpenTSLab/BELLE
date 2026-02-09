import torch

import math
import warnings

from belle.modules.optim import Eden
from torch.optim.lr_scheduler import LRScheduler


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed ** (-0.5) * min(
        step ** (-0.5), step * warmup_steps ** (-1.5)
    )


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        base_lr: float,
        optimizer: torch.optim.Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:

        self.dim_embed = dim_embed
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.base_lr * calc_lr(
            self._step_count, self.dim_embed, self.warmup_steps
        )
        return [lr] * self.num_param_groups

    def set_step(self, step: int):
        self._step_count = step

class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )

class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class WarmupLinearLR(LRScheduler):
    """Learning rate scheduler with warmup phase and linear decay to zero.
    
    The learning rate changes in three phases:
    1. Warmup phase: LR linearly increases from (lr * end_factor) to (lr * start_factor)
    2. Middle phase: LR linearly changes from (lr * start_factor) to (lr * end_factor)
    3. Final phase: LR linearly decreases to 0
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        start_factor (float): The multiplication factor after warmup phase.
            Default: 1.0.
        end_factor (float): The multiplication factor at the start of warmup
            and end of middle phase. Default: 0.1.
        total_iters (int): The number of iterations for the entire schedule.
            Default: 100.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    
    def __init__(self, optimizer, warmup_steps, start_factor=1.0, end_factor=0.1, 
                 total_iters=100, last_epoch=-1, verbose="deprecated"):
        if start_factor <= end_factor:
            raise ValueError('Starting factor expected to be greater than end factor.')
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError('Starting factor expected to be between 0 and 1.')
        if end_factor >= 1.0 or end_factor < 0:
            raise ValueError('End factor expected to be between 0 and 1.')
        if warmup_steps >= total_iters:
            raise ValueError('Warmup steps should be less than total iterations.')
            
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)
        
        # Determine the current phase and calculate the factor
        if self.last_epoch < self.warmup_steps:  # Warmup phase
            # Linear interpolation from end_factor to start_factor
            factor = self.end_factor + (self.start_factor - self.end_factor) * \
                    (self.last_epoch / self.warmup_steps)
        elif self.last_epoch < self.total_iters:  # Middle and final phase
            # Calculate remaining steps after warmup
            steps_after_warmup = self.last_epoch - self.warmup_steps
            remaining_steps = self.total_iters - self.warmup_steps
            # Linear interpolation from start_factor to end_factor
            factor = self.start_factor + (self.end_factor - self.start_factor) * \
                    (steps_after_warmup / remaining_steps)
        else:  # Beyond total_iters
            factor = self.end_factor
        
        # Apply the factor to all parameter groups
        return [group['initial_lr'] * factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_steps:  # Warmup phase
            factor = self.end_factor + (self.start_factor - self.end_factor) * \
                    (self.last_epoch / self.warmup_steps)
        elif self.last_epoch < self.total_iters:  # Middle and final phase
            steps_after_warmup = self.last_epoch - self.warmup_steps
            remaining_steps = self.total_iters - self.warmup_steps
            factor = self.start_factor + (self.end_factor - self.start_factor) * \
                    (steps_after_warmup / remaining_steps)
        else:  # Beyond total_iters
            factor = self.end_factor
            
        return [base_lr * factor for base_lr in self.base_lrs]
        

def get_scheduler(params, optimizer):
    if params.scheduler_name.lower() == "eden":
        scheduler = Eden(optimizer, 5000, 4, warmup_batches=params.num_epochs * params.steps_per_epoch * params.warmup_ratio)
    elif params.scheduler_name.lower() == "noam":
        scheduler = NoamScheduler(
            params.base_lr,
            optimizer,
            params.decoder_dim,
            warmup_steps=params.num_epochs * params.steps_per_epoch * params.warmup_ratio,
        )
        # scheduler.set_step(params.start_batch or params.batch_idx_train)
    elif params.scheduler_name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.num_epochs * params.steps_per_epoch, eta_min=0
        )
    elif params.scheduler_name.lower() == "linear":
        # scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=1.0, end_factor=0, total_iters=params.num_epochs * params.steps_per_epoch
        # )
        scheduler = WarmupLinearLR(
            optimizer, start_factor=1.0, end_factor=0, total_iters=params.num_epochs * params.steps_per_epoch, warmup_steps=params.num_epochs * params.steps_per_epoch * params.warmup_ratio
        )
    else:
        raise NotImplementedError(f"{params.scheduler_name}")

    return scheduler


if __name__ == "__main__":
    # test warmup linear lr
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt

    model = torch.nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print(optimizer.param_groups)
    scheduler = WarmupLinearLR(optimizer, warmup_steps=10, start_factor=1.0, end_factor=1e-4, total_iters=100)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=100)
    print(optimizer.param_groups)
    print(scheduler.get_last_lr())

    lrs = []
    for epoch in range(100):
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()
        lrs.append(lr)

    print(lrs)
    print(len(lrs))
    # plt.plot(lrs)
