import cupy as cp
import numpy as np

_grad_enabled = True

class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev

class Tensor:
    def __init__(self, data, _prev=(), requires_grad=False, device=None):
        if device == "cuda":
            self.backend = cp
            data = cp.array(data, dtype=cp.float32)
        elif device == "cpu":
            self.backend = np
            data = np.array(data, dtype=np.float32)
        else:
            if isinstance(data, cp.ndarray):
                self.backend = cp
                data = data.astype(cp.float32)
            elif isinstance(data, np.ndarray):
                self.backend = np
                data = data.astype(np.float32)
            else:
                self.backend = np
                data = np.array(data, dtype=np.float32)

        self.data = data
        self.requires_grad = requires_grad and _grad_enabled
        self.grad = self.backend.zeros_like(data) if self.requires_grad else None

        self._backward = lambda: None
        self._prev = set(_prev)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        axes = tuple(reversed(range(self.ndim)))
        return self.permute(*axes)

    def __add__(self, other):
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(out.grad, self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(other.data * out.grad, self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(self.data * out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data ** other.data, (self, other),requires_grad=requires_grad)

        def _backward():
            #with self.backend.errstate(divide='ignore', invalid='ignore'):
                Tensor._accumulate_grad(self, Tensor._unbroadcast((other.data * self.data**(other.data - 1)) * out.grad, self.data.shape))
                Tensor._accumulate_grad(other, Tensor._unbroadcast((out.data * self.backend.log(self.data)) * out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out_data = self.backend.matmul(self.data, other.data)
        out = Tensor(out_data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(self.backend.matmul(out.grad, other.data.T), self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(self.backend.matmul(self.data.T, out.grad), other.data.shape))
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Tensor._ensure_tensor(other, self.backend) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Tensor._ensure_tensor(other, self.backend) / self

    def __rpow__(self, other):
        return Tensor._ensure_tensor(other, self.backend) ** self

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = out.grad
            grad_full = self.backend.zeros_like(self.data)
            grad_full[idx] = grad
            Tensor._accumulate_grad(self, grad_full)

        out._backward = _backward
        return out

    def max(self, dim=None, keepdim=False):
        out_data = self.backend.max(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            od = out_data if (dim is None or keepdim) else self.backend.expand_dims(out.data, axis=dim)  # need to expand out_data to make it broadcastable with self.data
            mask = (self.data == od).astype(self.data.dtype)
            count = self.backend.sum(mask, axis=dim, keepdims=True)
            grad = (mask / count) * out.grad
            if not keepdim and dim is not None:
                grad = Tensor._expand_like(grad, self.data.shape, dim)
            Tensor._accumulate_grad(self, grad)
        out._backward = _backward

        return out

    def sum(self, dim=None, keepdim=False):
        out_data = self.backend.sum(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = out.grad
            if not keepdim and dim is not None:
                grad = Tensor._expand_like(grad, self.data.shape)
            Tensor._accumulate_grad(self, grad)
        out._backward = _backward

        return out

    def mean(self, dim=None, keepdim=False):
        out = self.sum(dim=dim, keepdim=keepdim)
        if dim is None:
            divisor = self.data.size
        else:
            divisor = self.data.shape[dim] if isinstance(dim, int) else self.backend.prod([self.data.shape[d] for d in dim])

        return out / self.backend.array(divisor, dtype=self.data.dtype)

    def var(self, dim=None, keepdim=False, unbiased=True):
        mean = self.mean(dim=dim, keepdim=True)
        sq_diff = (self - mean) ** 2
        out = sq_diff.sum(dim=dim, keepdim=keepdim)

        if dim is None:
            count = self.data.size
        else:
            count = self.data.shape[dim] if isinstance(dim, int) else self.backend.prod([self.data.shape[d] for d in dim])

        divisor = count - 1 if unbiased and count > 1 else count

        return out / self.backend.array(divisor, dtype=self.data.dtype)

    def reshape(self, *shape):
        out_data = self.data.reshape(*shape)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.grad.reshape(self.data.shape))
        out._backward = _backward

        return out

    def transpose(self, dim0, dim1):
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]

        return self.permute(*dims)

    def permute(self, *dims):
        out_data = self.data.transpose(*dims)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            reverse_dims = np.argsort(dims) if dims else None
            Tensor._accumulate_grad(self, out.grad.transpose(*reverse_dims))
        out._backward = _backward

        return out

    def squeeze(self, dim=None):
        out_data = self.backend.squeeze(self.data, dim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.grad.reshape(self.shape))
        out._backward = _backward

        return out

    def unsqueeze(self, dim):
        out_data = self.backend.expand_dims(self.data, axis=dim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, self.backend.squeeze(out.grad, axis=dim))

        out._backward = _backward

        return out

    def gather(self, dim, index):
        out_data = self.backend.take_along_axis(self.data, index.data, axis=dim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = self.backend.zeros_like(self.data)
            self.backend.put_along_axis(grad, index.data, out.grad, axis=dim)
            Tensor._accumulate_grad(self, grad)

        out._backward = _backward

        return out

    def exp(self):
        out_data = self.backend.exp(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.data * out.grad)
        out._backward = _backward

        return out

    def log(self):
        out_data = self.backend.log(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            #with self.backend.errstate(divide='ignore', invalid='ignore'):
                Tensor._accumulate_grad(self, out.grad / self.data)
        out._backward = _backward

        return out

    def tanh(self):
        out_data = self.backend.tanh(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (1 - out.data**2) * out.grad)
        out._backward = _backward

        return out

    def sigmoid(self):
        out_data = 1 / (1 + self.backend.exp(-self.data))
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (out.data - out.data**2) * out.grad)
        out._backward = _backward

        return out

    def relu(self):
        out_data = self.backend.maximum(0, self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (self.data > 0).astype(self.data.dtype) * out.grad)
        out._backward = _backward

        return out

    def gelu(self):
        """
        Approximate implementation of GeLU, corresponds to torch.nn.functional.gelu with approximate="tanh"
        """
        c = (2 / self.backend.pi) ** 0.5
        return 0.5 * self * (1 + ((self + 0.044715 * self ** 3) * c).tanh())
    
    def logsumexp(self, dim=None, keepdim=False):
        max_val = self.max(dim=dim, keepdim=True)

        shifted = self - max_val
        exp_shifted = shifted.exp()

        sum_exp = exp_shifted.sum(dim=dim, keepdim=keepdim)
        log_sum_exp = sum_exp.log()

        if keepdim:
            return log_sum_exp + max_val
        else:
            return log_sum_exp + max_val.squeeze(dim)

    def log_softmax(self, dim=None):
        return self - self.logsumexp(dim=dim, keepdim=True)
    
    def pad2d(self, padding):
        """
        Zero-pad H and W. padding can be:
        - int p      -> pad (p,p,p,p)
        - (pH, pW)   -> pad (pH,pH,pW,pW)
        - (t, b, l, r)
        """
        if isinstance(padding, int):
            t = b = l = r = padding
        elif len(padding) == 2:
            t = b = int(padding[0]); l = r = int(padding[1])
        else:
            t, b, l, r = map(int, padding)

        xp = self.backend
        out_data = xp.pad(self.data, ((0,0),(0,0),(t,b),(l,r)), mode="constant")
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None: return
            g = out.grad[:, :, t:t+self.shape[2], l:l+self.shape[3]]
            Tensor._accumulate_grad(self, g)

        out._backward = _backward
        return out

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradient")
        if gradient is None:
            self.grad = self.backend.ones_like(self.data)
        else:
            self.grad = self.backend.array(gradient, dtype=self.data.dtype)

        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            if t.requires_grad:
                t._backward()

    def zero_grad(self):
        if self.requires_grad:
          self.grad = self.backend.zeros_like(self.data)

    def __repr__(self):
        """
        Format tensor content, for device info cupy always uses CUDA
        """
        data_str = self.backend.array2string(self.data, separator=', ', prefix='tensor(')

        details = [f"dtype={self.data.dtype}, requires_grad={self.requires_grad}"]

        try:
            device_str = f"cuda:{self.data.device.id}"
            details.append(f"device='{device_str}'")
        except Exception:
            pass

        return f"tensor({data_str}, {', '.join(details)})"

    def to(self, device):
        if device == "cpu" and self.backend is cp:
            self.data = cp.asnumpy(self.data)
            self.grad = cp.asnumpy(self.grad) if self.grad is not None else None
            self.backend = np
        elif device == "cuda" and self.backend is np:
            self.data = cp.array(self.data)
            self.grad = cp.array(self.grad) if self.grad is not None else None
            self.backend = cp
        return self

    def xp(self):
        return self.backend
    
    @staticmethod
    def cat(tensors, dim=0):
        assert len(tensors) > 0
        backend = tensors[0].backend
        data = backend.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, _prev=tuple(tensors), requires_grad=requires_grad)

        def _backward():
            if out.grad is None: return
            sizes = [t.shape[dim] for t in tensors]
            start = 0
            for t, sz in zip(tensors, sizes):
                slc = [slice(None)] * out.grad.ndim
                slc[dim] = slice(start, start + sz)
                Tensor._accumulate_grad(t, out.grad[tuple(slc)])
                start += sz

        out._backward = _backward
        return out

    @staticmethod
    def _im2col(x, kH, kW, sH, sW, dH, dW, Hout, Wout, pH, pW):
        """
        x: (N, C, H, W) Tensor
        returns Xcols: (N, C*kH*kW, Hout*Wout) Tensor
        """
        x_pad = x.pad2d((pH, pH, pW, pW))

        cols = []
        for ky in range(kH):
            y0 = ky * dH
            y1 = y0 + sH * Hout
            for kx in range(kW):
                x0 = kx * dW
                x1 = x0 + sW * Wout
                patch = x_pad[:, :, y0:y1:sH, x0:x1:sW]
                cols.append(patch.reshape(x.shape[0], x.shape[1], -1))
        Xcols = Tensor.cat(cols, dim=1)
        return Xcols

    @staticmethod
    def _expand_like(x, target_shape, dims_reduced):
        """
        Expand tensor x to match target_shape by adding dimensions at dims_reduced
        """
        if isinstance(dims_reduced, int):
            dims_reduced = (dims_reduced,)

        backend = x.backend

        for dim in sorted(dims_reduced):
            x = backend.expand_dims(x, axis=dim)

        return backend.broadcast_to(x, target_shape)

    @staticmethod
    def _unbroadcast(x, target_shape):
        """
        Reduces shape of tensor x back to target_shape by summing over broadcasted dimensions
        """
        while x.ndim > len(target_shape):
            x = x.sum(axis=0)
        for i, (g, t) in enumerate(zip(x.shape, target_shape)):
            if g != t:
                x = x.sum(axis=i, keepdims=True)

        return x.reshape(target_shape)

    @staticmethod
    def _ensure_tensor(x, backend):
        return x if isinstance(x, Tensor) else Tensor(x, device="cuda" if backend is cp else "cpu")

    @staticmethod
    def _accumulate_grad(tensor, grad):
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad

    @staticmethod
    def zeros(*shape, requires_grad=False, device="cpu"):
        xp = np if device == "cpu" else cp
        data = xp.zeros(shape, dtype=xp.float32)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False, scale=1., device="cpu"):
        xp = np if device == "cpu" else cp
        data = scale * xp.random.randn(*shape).astype(xp.float32)
        return Tensor(data, requires_grad=requires_grad)