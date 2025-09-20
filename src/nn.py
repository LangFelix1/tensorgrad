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
    
    def state_dict(self):
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data.copy()

        for name, module in self._modules.items():
            sub_state = module.state_dict()
            for sub_name, value in sub_state.items():
                state[f"{name}.{sub_name}"] = value

        return state

    def load_state_dict(self, state_dict):
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data[:] = state_dict[name]
            else:
                raise KeyError(f"{name} not found in state_dict")

        for name, module in self._modules.items():
            sub_state = {
                k[len(name) + 1:]: v
                for k, v in state_dict.items()
                if k.startswith(f"{name}.")
            }
            module.load_state_dict(sub_state)

    def to(self, device):
        for name, param in self._parameters.items():
            param.to(device)
        for module in self._modules.values():
            module.to(device)
        return self
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        gain = (2. / in_features) ** 0.5
        self.weight = Tensor.randn(out_features, in_features, requires_grad=True, scale=gain)
        self.bias = Tensor.zeros(out_features, requires_grad=True) if bias else None

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
        log_probs = logits.log_softmax(dim=1)                         # shape: (B, C)
        targets = targets.unsqueeze(1)                                # shape: (B, 1)
        picked_log_probs = log_probs.gather(dim=1, index = targets)   # shape: (B, 1)
        loss = -picked_log_probs.squeeze(1)                           # shape: (B,)

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        
class Dropout(Module):
    # potential problem due to computational graph and device mismatching
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, x):
        if self.training:
            backend = x.backend
            mask = (backend.random.rand(*x.shape) > self.p).astype(x.data.dtype)
            return x * Tensor(mask, requires_grad=False) / (1 - self.p)
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
            self.weight = Tensor.ones(num_features, requires_grad=True)
            self.bias = Tensor.zeros(num_features, requires_grad=True)
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor.zeros(num_features)
            self.running_var = Tensor.ones(num_features)

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
        
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.has_bias = bias

        if elementwise_affine:
            self.weight = Tensor.ones(*self.normalized_shape, requires_grad=True)
            self.bias = Tensor.zeros(*self.normalized_shape, requires_grad=True) if bias else None
        else:
            self.weight = None
            self.bias = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, bias={self.has_bias})"

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)

        if self.elementwise_affine:
            x_hat = x_hat * self.weight
            if self.bias is not None:
                x_hat = x_hat + self.bias
        return x_hat
    
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.randn(num_embeddings, embedding_dim, requires_grad=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"

    def forward(self, indices):
        backend = self.weight.backend
        embedded_data = self.weight.data[indices.data.astype(backend.int64)]
        out = Tensor(embedded_data, _prev=(self.weight,), requires_grad=self.weight.requires_grad)

        def _backward():
            grad_output = out.grad
            backend = self.weight.backend
            flat_indices = indices.data.ravel().astype(backend.int64)
            flat_grads = grad_output.reshape(-1, self.embedding_dim)
            grad_weight = backend.zeros_like(self.weight.data)
            backend.add.at(grad_weight, flat_indices, flat_grads)

            Tensor._accumulate_grad(self.weight, grad_weight)

        out._backward = _backward
        return out

class Conv2d(Module):
    """
    Minimal NCHW Conv2d with stride, padding, dilation, bias (no groups).
    PyTorch-like shapes:
      input:  (N, C_in, H, W)
      weight: (C_out, C_in, kH, kW)
      bias:   (C_out,)
      output: (N, C_out, H_out, W_out)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        kH, kW = self._to_pair(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kH, kW)
        self.stride = self._to_pair(stride)
        self.padding = self._to_pair(padding)
        self.dilation = self._to_pair(dilation)

        # Kaiming fan_in init (like PyTorch default for Conv2d)
        fan_in = in_channels * kH * kW
        scale = (2.0 / fan_in) ** 0.5
        self.weight = Tensor.randn(out_channels, in_channels, kH, kW, requires_grad=True, scale=scale)
        self.bias = Tensor.zeros(out_channels, requires_grad=True) if bias else None

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"bias={self.bias is not None})")

    @staticmethod
    def _to_pair(v):
        return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))

    @staticmethod
    def _out_hw(H, W, kH, kW, pH, pW, sH, sW, dH, dW):
        eff_kH = (kH - 1) * dH + 1
        eff_kW = (kW - 1) * dW + 1
        Hout = (H + 2 * pH - eff_kH) // sH + 1
        Wout = (W + 2 * pW - eff_kW) // sW + 1
        return int(Hout), int(Wout)

    @staticmethod
    def _im2col(xp, x_pad, kH, kW, sH, sW, dH, dW, Hout, Wout):
        # x_pad: (N, C, Hpad, Wpad)
        N, C, Hp, Wp = x_pad.shape
        cols = xp.empty((N, C * kH * kW, Hout * Wout), dtype=x_pad.dtype)
        row = 0
        for ky in range(kH):
            y0 = ky * dH
            y1 = y0 + sH * Hout
            for kx in range(kW):
                x0 = kx * dW
                x1 = x0 + sW * Wout
                patch = x_pad[:, :, y0:y1:sH, x0:x1:sW]          # (N, C, Hout, Wout)
                cols[:, row * C:(row + 1) * C, :] = patch.reshape(N, C, -1)
                row += 1
        return cols  # (N, C*kH*kW, Hout*Wout)

    @staticmethod
    def _col2im(xp, cols, N, C, H, W, kH, kW, pH, pW, sH, sW, dH, dW, Hout, Wout):
        Hpad, Wpad = H + 2 * pH, W + 2 * pW
        xg_pad = xp.zeros((N, C, Hpad, Wpad), dtype=cols.dtype)
        row = 0
        for ky in range(kH):
            y0 = ky * dH
            y1 = y0 + sH * Hout
            for kx in range(kW):
                x0 = kx * dW
                x1 = x0 + sW * Wout
                patch = cols[:, row * C:(row + 1) * C, :].reshape(N, C, Hout, Wout)
                xg_pad[:, :, y0:y1:sH, x0:x1:sW] += patch
                row += 1
        return xg_pad[:, :, pH:pH + H, pW:pW + W]

    def forward(self, x: Tensor):
        xp = x.backend
        N, C, H, W = x.shape
        assert C == self.in_channels, f"in_channels={self.in_channels} but got input with C={C}"

        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        Hout, Wout = self._out_hw(H, W, kH, kW, pH, pW, sH, sW, dH, dW)
        if Hout <= 0 or Wout <= 0:
            raise ValueError("Invalid output size; check stride/padding/dilation/kernel_size.")

        # pad input and im2col
        x_pad = xp.pad(x.data, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode="constant")
        Xcols = self._im2col(xp, x_pad, kH, kW, sH, sW, dH, dW, Hout, Wout)     # (N, C*kH*kW, Hout*Wout)
        Wcol = self.weight.data.reshape(self.out_channels, -1)                   # (Cout, C*kH*kW)

        # batched GEMM: (N, Cout, Hout*Wout)
        Ymat = xp.matmul(Wcol[None, :, :], Xcols)
        if self.bias is not None:
            Ymat += self.bias.data.reshape(1, -1, 1)
        y_data = Ymat.reshape(N, self.out_channels, Hout, Wout)

        requires_grad = x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        out = Tensor(y_data, _prev=(x, self.weight) + ((self.bias,) if self.bias is not None else ()),
                     requires_grad=requires_grad)

        def _backward():
            dY = out.grad                                # (N, Cout, Hout, Wout)
            if dY is None:
                return
            dYmat = dY.reshape(N, self.out_channels, Hout * Wout)

            # dW: sum over batch
            # (N, Cout, Hout*Wout) @ (N, Hout*Wout, C*kH*kW) -> (N, Cout, C*kH*kW) -> sum over N
            dWcol = xp.matmul(dYmat, Xcols.transpose(0, 2, 1)).sum(axis=0)      # (Cout, C*kH*kW)
            dW = dWcol.reshape(self.weight.data.shape)

            # dB
            if self.bias is not None:
                dB = dYmat.sum(axis=(0, 2))                                     # (Cout,)

            # dX via W^T * dY
            dXcols = xp.matmul(Wcol.T[None, :, :], dYmat)                        # (N, C*kH*kW, Hout*Wout)
            dX = self._col2im(xp, dXcols, N, C, H, W, kH, kW, pH, pW, sH, sW, dH, dW, Hout, Wout)

            Tensor._accumulate_grad(self.weight, dW)
            if self.bias is not None:
                Tensor._accumulate_grad(self.bias, dB.reshape(self.bias.data.shape))
            Tensor._accumulate_grad(x, dX)

        out._backward = _backward
        return out