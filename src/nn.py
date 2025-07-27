import cupy as cp
from src.tensor import Tensor

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def __repr__(self):
        modstr = self._modules.items()
        lines = [f"{self.__class__.__name__}("]
        for name, module in modstr:
            mod_repr = repr(module)
            mod_repr = "\n    ".join(mod_repr.splitlines())
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return "\n".join(lines)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Tensor(cp.random.randn(out_features, in_features) * cp.sqrt(2. / in_features), requires_grad=True)
        self.bias = Tensor(cp.zeros(out_features), requires_grad=True) if bias else None

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None})"

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
class ReLU(Module):
    def __repr__(self):
      return f"{self.__class__.__name__}()"

    def forward(self, x):
        return x.relu()
    
class Tanh(Module):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, x):
        return x.tanh()
    
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules_list = []

        for idx, module in enumerate(modules):
            assert isinstance(module, Module) #, f"All elements must be Module instances, got {type(module)}"
            self._modules_list.append(module)
            self._modules[str(idx)] = module

    def forward(self, x):
        for module in self._modules_list:
            x = module(x)
        return x

    def __getitem__(self, idx):
        return self._modules_list[idx]
    
class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction})"

    def forward(self, input, target):
        diff = input - target
        loss = diff * diff

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        
class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction})"

    def forward(self, logits, targets):
        """
        logits: Tensor of shape (B, C)
        targets: Tensor of shape (B,) with integer class indices
        """
        log_probs = logits.log_softmax(dim=1)                 # shape: (B, C)
        targets = targets.unsqueeze(1)                        # shape: (B, 1)
        picked_log_probs = log_probs.gather(targets, dim=1)   # shape: (B, 1)
        loss = -picked_log_probs.squeeze(1)                   # shape: (B,)

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, x):
        if self.training:
            mask = (cp.random.rand(*x.shape) > self.p).astype(x.data.dtype)
            return x * Tensor(mask) / (1 - self.p)
        else:
            return x
        
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Tensor(cp.ones(num_features), requires_grad=True)
            self.bias = Tensor(cp.zeros(num_features), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor(cp.zeros(num_features))
            self.running_var = Tensor(cp.ones(num_features))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"

    def forward(self, x):
        if self.training:
            reduce_dims = tuple(i for i in range(x.ndim) if i != 1)
            mean = x.mean(dim=reduce_dims, keepdim=True)
            var = x.var(dim=reduce_dims, keepdim=True, unbiased=False)

            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.reshape((1, -1) + (1,) * (x.ndim - 2))
            var = self.running_var.reshape((1, -1) + (1,) * (x.ndim - 2))

        x_hat = (x - mean) / ((var + self.eps) ** 0.5)

        if self.affine:
            w = self.weight.reshape((1, -1) + (1,) * (x.ndim - 2))
            b = self.bias.reshape((1, -1) + (1,) * (x.ndim - 2))
            return x_hat * w + b
        else:
            return x_hat