import cupy as cp
from tensor import Tensor

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

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