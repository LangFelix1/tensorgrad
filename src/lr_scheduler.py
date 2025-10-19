import numpy as np

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self, *args, **kwargs):
        raise NotImplementedError
    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}
    def load_state_dict(self, state): self.__dict__.update(state)

class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=0):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=0):
        super().__init__(optimizer)
        self.milestones = sorted(m for m in milestones)
        self.gamma = gamma
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            self.optimizer.lr *= self.gamma

class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=0):
        super().__init__(optimizer)
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self.last_epoch)

class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        t = min(self.last_epoch, self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * t / self.T_max))

class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=5, threshold=1e-4, min_lr=0.0, cooldown=0):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.best = None
        self.bad = 0
        self.cool = 0

    def step(self, metric):
        if self.cool > 0:
            self.cool -= 1
        improved = (self.best is None) or (metric < self.best - self.threshold)
        if improved:
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
            if self.bad > self.patience and self.cool == 0:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    self.optimizer.lr = new_lr
                    self.cool = self.cooldown
                self.bad = 0

class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0.0, lr_warmup_start=1e-8):
        super().__init__(optimizer)
        self.base_lr = optimizer.lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        self.lr_warmup_start = lr_warmup_start
        self.t = 0

    def step(self):
        self.t += 1
        if self.t <= self.warmup_steps:
            a = (self.base_lr - self.lr_warmup_start) / max(1, self.warmup_steps)
            self.optimizer.lr = self.lr_warmup_start + a * self.t
        else:
            tw = self.t - self.warmup_steps
            T = max(1, self.T_max - self.warmup_steps)
            self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * min(tw, T) / T))