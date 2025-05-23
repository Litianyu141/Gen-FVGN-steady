import matplotlib.pyplot as plt
from torch import nn
import os
from torch import optim
from math import pow
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class GradualStepExplrScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        milestone: same as torch.optim.lr_scheduler.MultiStepLR
        gamma : return base_lr*self.gamma**self.milestone_hit
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(
        self,
        optimizer,
        multiplier,
        milestone: list,
        gamma,
        after_scheduler,
        expgamma,
        total_epoch,
        decay_steps,
        min_lr=1e-6,
        last_epoch=-1,
    ):
        self.multiplier = multiplier
        for milestone_steps in milestone:
            if milestone_steps > total_epoch:
                raise ValueError(
                    "steps in milestone should be smaller than total_epoch"
                )
        self.milestone = milestone
        self.gamma = gamma
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self._last_step_lr = None
        self.milestone_hit = 0
        self.milestone_trigger = True
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.expgamma = expgamma
        self.decay_steps = decay_steps

        super(GradualStepExplrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            return [
                min_lr
                + max(base_lr * self.gamma**self.milestone_hit - min_lr, 0)
                * pow(
                    self.expgamma,
                    ((self.last_epoch - self.total_epoch) / self.decay_steps),
                )
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
            ]

        else:
            if self.last_epoch in self.milestone:
                self.milestone_hit += 1
                self._last_step_lr = [
                    base_lr * self.gamma**self.milestone_hit
                    for base_lr in self.base_lrs
                ]
                return self._last_step_lr
            else:
                self._last_step_lr = [
                    base_lr * self.gamma**self.milestone_hit
                    for base_lr in self.base_lrs
                ]
                return self._last_step_lr

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        print("warmuping...")
        if self.last_epoch <= self.total_epoch:
            warmup_lr = None
            if self.multiplier == 1.0:
                warmup_lr = [
                    base_lr * (float(self.last_epoch) / self.total_epoch)
                    for base_lr in self.base_lrs
                ]
            else:
                warmup_lr = [
                    base_lr
                    * (
                        (self.multiplier - 1.0) * self.last_epoch / self.total_epoch
                        + 1.0
                    )
                    for base_lr in self.base_lrs
                ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            return super(GradualStepExplrScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def get_variable(self):
        dict = {
            "_last_step_lr": self._last_step_lr,
            "last_epoch": self.last_epoch,
            "finished": self.finished,
            "milestone_hit": self.milestone_hit,
            "min_lrs": self.min_lrs,
        }
        return dict


class ExpLR(_LRScheduler):
    """
    Exponential learning rate scheduler
    Based on procedure described in LTS
    If min_lr==0 and decay_steps==1, same as torch.optim.lr_scheduler.ExpLR
    """

    def __init__(
        self, optimizer, decay_steps=10000, gamma=0.4, min_lr=1e-7, last_epoch=-1
    ):
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.gamma = gamma
        self.decay_steps = decay_steps

        super(ExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            min_lr
            + max(base_lr - min_lr, 0)
            * pow(self.gamma, self.last_epoch / self.decay_steps)
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Linear(10, 10)

    def forward(self, input):
        out = self.net(input)
        return out

class StepexpLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        startlr,
        steplr_milestone,
        steplr_gamma,
        explr_milestone,
        explr_gamma,
        total_epoch,
        min_lr,
        last_epoch=-1,
    ):
        self.startlr = startlr
        self.steplr_milestone = steplr_milestone
        self.steplr_gamma = steplr_gamma
        self.explr_milestone = explr_milestone
        self.explr_gamma = explr_gamma
        self.total_epoch = total_epoch

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.decay_steps = total_epoch - explr_milestone
        super(StepexpLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.steplr_milestone:
            return [self.startlr for _ in self.base_lrs]
        elif self.steplr_milestone <= self.last_epoch < self.explr_milestone:
            return [self.startlr * self.steplr_gamma for _ in self.base_lrs]
        else:
            decay_progress = (self.last_epoch - self.explr_milestone) / self.decay_steps
            return [
                min_lr
                + max(base_lr * self.steplr_gamma - min_lr, 0)
                * pow(self.explr_gamma, decay_progress)
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
            ]

"""Test Lr curve and plot"""
if __name__ == "__main__":
    
    saving_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f"{saving_path}/lr_vis", exist_ok=True)
    
    ver="new" 
    model = Net()
    startlr = 1e-4
    total_epoch = 100000
    optimizer = optim.Adam(model.parameters(), lr=startlr)

    steplr_decay_steps = int(total_epoch* 0.1)
    explr_decay_steps = int(total_epoch * 0.3)
    if ver == "old":
        two_step_scheduler = ExpLR(
            optimizer, decay_steps=total_epoch - explr_decay_steps, gamma=1e-4
        )
        scheduler = GradualStepExplrScheduler(
            optimizer,
            multiplier=1.0,
            milestone=[steplr_decay_steps],
            gamma=0.1,
            total_epoch=explr_decay_steps,
            after_scheduler=two_step_scheduler,
            expgamma=1e-2,
            decay_steps=total_epoch - explr_decay_steps,
            min_lr=1e-6,
        )
    elif ver == "new":
            
        scheduler = StepexpLRScheduler(
            optimizer = optimizer,
            startlr=startlr,
            steplr_milestone=steplr_decay_steps,
            steplr_gamma=1,
            explr_milestone=explr_decay_steps,
            explr_gamma=0.1,
            total_epoch=total_epoch,
            min_lr=1e-6,
        )

    # Collect learning rates and write to .dat file
    dat_file_path = f"{saving_path}/lr_vis/learning_rate_schedule_{ver}.dat"
    with open(dat_file_path, 'w') as f:
        f.write('Variables="epoch" "learning_rate"\n')
        
        lrs = []
        for epoch in range(total_epoch):
            lr = scheduler.get_lr()[0]  # Get the first (and only) learning rate
            lrs.append(lr)
            f.write(f"{epoch} {lr:.10f}\n")
            
            optimizer.step()
            scheduler.step()

    print(f"Learning rate schedule written to {dat_file_path}")

    # Plot the learning rate curve
    plt.figure(figsize=(12, 6))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    # plt.yscale('log')
    plt.axvline(x=steplr_decay_steps, color='r', linestyle='--', label='Step LR Milestone')
    plt.axvline(x=explr_decay_steps, color='g', linestyle='--', label='Exp LR Milestone')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{saving_path}/lr_vis/learning_rate_schedule_{ver}.png')

