class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self, *args, **kwargs):
        raise NotImplementedError
    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}
    def load_state_dict(self, state): self.__dict__.update(state)