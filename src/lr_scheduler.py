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
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.last_epoch = int(last_epoch)

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=0):
        super().__init__(optimizer)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)
        self.last_epoch = int(last_epoch)

    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            self.optimizer.lr *= self.gamma

class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=0):
        super().__init__(optimizer)
        self.gamma = float(gamma)
        self.base_lr = optimizer.lr
        self.last_epoch = int(last_epoch)

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self.last_epoch)