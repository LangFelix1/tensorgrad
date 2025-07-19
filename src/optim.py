class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.zero_grad()

    def step(self):
        raise NotImplementedError