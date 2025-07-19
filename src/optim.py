class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.zero_grad()

    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0., dampening=0., weight_decay=0., nesterov=False):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.state = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            d_p = p.grad.copy()

            if self.weight_decay > 0:
                d_p += self.weight_decay * p.data

            if self.momentum > 0:
                buf = self.state.get(p)
                if buf is None:
                    buf = d_p.copy()
                else:
                    buf *= self.momentum
                    buf += (1 - self.dampening) * d_p
                self.state[p] = buf

                if self.nesterov:
                    d_p += self.momentum * buf
                else:
                    d_p = buf

            p.data -= self.lr * d_p