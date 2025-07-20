import cupy as cp

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

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=False):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            if p not in self.state:
                self.state[p] = {
                    "step": 0,
                    "exp_avg": cp.zeros_like(p.data),
                    "exp_avg_sq": cp.zeros_like(p.data),
                }
                if self.amsgrad:
                    self.state[p]["max_exp_avg_sq"] = cp.zeros_like(p.data)

            state = self.state[p]
            d_p = p.grad.copy()

            if self.weight_decay != 0:
                d_p += self.weight_decay * p.data

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            beta1, beta2 = self.betas

            state["step"] += 1
            step = state["step"]

            exp_avg[:] = beta1 * exp_avg + (1 - beta1) * d_p            # exponential moving average of the gradient (m_t)
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * d_p**2   # exponential moving average of the squared gradient (v_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            exp_avg_hat = exp_avg / bias_correction1
            if self.amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
                cp.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = cp.sqrt(max_exp_avg_sq / bias_correction2) + self.eps
            else:
                denom = cp.sqrt(exp_avg_sq / bias_correction2) + self.eps

            p.data -= self.lr * exp_avg_hat / denom

class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            if p not in self.state:
                self.state[p] = {
                    "step": 0,
                    "exp_avg": cp.zeros_like(p.data),
                    "exp_avg_sq": cp.zeros_like(p.data),
                }
                if self.amsgrad:
                    self.state[p]["max_exp_avg_sq"] = cp.zeros_like(p.data)

            state = self.state[p]
            d_p = p.grad.copy()

            p.data *= (1 - self.lr * self.weight_decay)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            beta1, beta2 = self.betas

            state["step"] += 1
            step = state["step"]

            exp_avg[:] = beta1 * exp_avg + (1 - beta1) * d_p            # exponential moving average of the gradient (m_t)
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * d_p**2   # exponential moving average of the squared gradient (v_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            exp_avg_hat = exp_avg / bias_correction1
            if self.amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
                cp.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = cp.sqrt(max_exp_avg_sq / bias_correction2) + self.eps
            else:
                denom = cp.sqrt(exp_avg_sq / bias_correction2) + self.eps

            p.data -= self.lr * exp_avg_hat / denom