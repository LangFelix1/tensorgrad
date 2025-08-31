import src.backend as backend

xp = backend.backend()

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.zero_grad()

    def step(self):
        raise NotImplementedError
    
    def state_dict(self):
        return {
            "hyperparams": self._get_hyperparams(),
            "state": [
                self._serialize_param_state(p) if p in self.state else None
                for p in self.params
            ],
        }

    def load_state_dict(self, state_dict):
        self._set_hyperparams(state_dict["hyperparams"])
        self.state = {}
        for p, s in zip(self.params, state_dict["state"]):
            if s is not None:
                self.state[p] = self._deserialize_param_state(p, s)

    def _get_hyperparams(self):
        raise NotImplementedError

    def _set_hyperparams(self, hyperparams):
        raise NotImplementedError

    def _serialize_param_state(self, p):
        raise NotImplementedError

    def _deserialize_param_state(self, state):
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

    def _get_hyperparams(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "dampening": self.dampening,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }
    
    def _set_hyperparams(self, hyperparams):
        self.lr = hyperparams["lr"]
        self.momentum = hyperparams["momentum"]
        self.dampening = hyperparams["dampening"]
        self.weight_decay = hyperparams["weight_decay"]
        self.nesterov = hyperparams["nesterov"]
    
    def _serialize_param_state(self, p):
        buf = self.state.get(p)
        return {
            "momentum_buffer": buf.copy() if buf is not None else None
        }
    
    def _deserialize_param_state(self, p, state):
        if state["momentum_buffer"] is not None:
            self.state[p] = state["momentum_buffer"].copy()

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
                    "exp_avg": xp.zeros_like(p.data),
                    "exp_avg_sq": xp.zeros_like(p.data),
                }
                if self.amsgrad:
                    self.state[p]["max_exp_avg_sq"] = xp.zeros_like(p.data)

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
                xp.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = xp.sqrt(max_exp_avg_sq / bias_correction2) + self.eps
            else:
                denom = xp.sqrt(exp_avg_sq / bias_correction2) + self.eps

            p.data -= self.lr * exp_avg_hat / denom

    def _get_hyperparams(self):
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }
    
    def _set_hyperparams(self, hyperparams):
        self.lr = hyperparams["lr"]
        self.betas = tuple(hyperparams["betas"])
        self.eps = hyperparams["eps"]
        self.weight_decay = hyperparams["weight_decay"]
        self.amsgrad = hyperparams["amsgrad"]

    def _serialize_param_state(self, p):
        s = self.state[p]
        result = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and "max_exp_avg_sq" in s:
            result["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()
        return result
    
    def _deserialize_param_state(self, p, s):
        self.state[p] = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and s.get("max_exp_avg_sq") is not None:
            self.state[p]["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()

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
                    "exp_avg": xp.zeros_like(p.data),
                    "exp_avg_sq": xp.zeros_like(p.data),
                }
                if self.amsgrad:
                    self.state[p]["max_exp_avg_sq"] = xp.zeros_like(p.data)

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
                xp.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = xp.sqrt(max_exp_avg_sq / bias_correction2) + self.eps
            else:
                denom = xp.sqrt(exp_avg_sq / bias_correction2) + self.eps

            p.data -= self.lr * exp_avg_hat / denom

    def _get_hyperparams(self):
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }
    
    def _set_hyperparams(self, hyperparams):
        self.lr = hyperparams["lr"]
        self.betas = tuple(hyperparams["betas"])
        self.eps = hyperparams["eps"]
        self.weight_decay = hyperparams["weight_decay"]
        self.amsgrad = hyperparams["amsgrad"]

    def _serialize_param_state(self, p):
        s = self.state[p]
        result = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and "max_exp_avg_sq" in s:
            result["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()
        return result
    
    def _deserialize_param_state(self, p, s):
        self.state[p] = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and s.get("max_exp_avg_sq") is not None:
            self.state[p]["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()