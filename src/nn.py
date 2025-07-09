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