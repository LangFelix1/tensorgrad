import cupy as cp

class Tensor:
    def __init__(self, data, _prev=(), requires_grad=False):
        if not isinstance(data, cp.ndarray):
            data = cp.array(data, dtype=cp.float32)
        else:
            data = data.astype(cp.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = cp.zeros_like(data) if requires_grad else None

        self._backward = lambda: None
        self._prev = set(_prev)

    def __add__(self, other):
        other = Tensor._ensure_tensor(other)

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
        other = Tensor._ensure_tensor(other)

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
        other = Tensor._ensure_tensor(other)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data ** other.data, (self, other),requires_grad=requires_grad)

        def _backward():
            with cp.errstate(divide='ignore', invalid='ignore'):
                Tensor._accumulate_grad(self, Tensor._unbroadcast((other.data * self.data**(other.data - 1)) * out.grad, self.data.shape))
                Tensor._accumulate_grad(other, Tensor._unbroadcast((out.data * cp.log(self.data)) * out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = Tensor._ensure_tensor(other)

        requires_grad = self.requires_grad or other.requires_grad
        out_data = cp.matmul(self.data, other.data)
        out = Tensor(out_data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(cp.matmul(out.grad, other.data.T), self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(cp.matmul(self.data.T, out.grad), other.data.shape))
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Tensor._ensure_tensor(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Tensor._ensure_tensor(other) / self

    def __rpow__(self, other):
        return Tensor._ensure_tensor(other) ** self

    def sum(self, dim=None, keepdim=False):
        out_data = cp.sum(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = out.grad
            if not keepdim and dim is not None:
                grad = Tensor._expand_like(grad, self.data.shape, dim)
            Tensor._accumulate_grad(self, grad)
        out._backward = _backward

        return out

    def mean(self, dim=None, keepdim=False):
        out = self.sum(dim=dim, keepdim=keepdim)
        if dim is None:
            divisor = self.data.size
        else:
            divisor = self.data.shape[dim] if isinstance(dim, int) else cp.prod([self.data.shape[d] for d in dim])

        return out / cp.array(divisor, dtype=self.data.dtype)

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
            reverse_dims = cp.argsort(dims) if dims else None
            Tensor._accumulate_grad(self, out.grad.transpose(*reverse_dims))
        out._backward = _backward

        return out

    def exp(self):
        out_data = cp.exp(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.data * out.grad)
        out._backward = _backward

        return out

    def log(self):
        out_data = cp.log(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            with cp.errstate(divide='ignore', invalid='ignore'):
                Tensor._accumulate_grad(self, out.grad / self.data)
        out._backward = _backward

        return out

    def tanh(self):
        out_data = cp.tanh(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (1 - out.data**2) * out.grad)
        out._backward = _backward

        return out

    def sigmoid(self):
        out_data = 1 / (1 + cp.exp(-self.data))
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (out.data - out.data**2) * out.grad)
        out._backward = _backward

        return out

    def relu(self):
        out_data = cp.maximum(0, self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (self.data > 0).astype(self.data.dtype) * out.grad)
        out._backward = _backward

        return out

    def gelu(self):
        """
        Approximate implementation of GeLU, corresponds to torch.nn.functional.gelu with approximate="tanh"
        """
        c = cp.sqrt(2 / cp.pi)
        return 0.5 * self * (1 + ((self + 0.044715 * self ** 3) * c).tanh())

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradient")
        if gradient is None:
            self.grad = cp.ones_like(self.data)
        else:
            self.grad = cp.array(gradient, dtype=self.data.dtype)

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
          self.grad = cp.zeros_like(self.data)

    def __repr__(self):
        """
        Format tensor content, for device info cupy always uses CUDA
        """
        data_str = cp.array2string(self.data, separator=', ', prefix='tensor(')

        details = [f"dtype={self.data.dtype}, requires_grad={self.requires_grad}"]

        try:
            device_str = f"cuda:{self.data.device.id}"
            details.append(f"device='{device_str}'")
        except Exception:
            pass

        return f"tensor({data_str}, {', '.join(details)})"

    @staticmethod
    def _expand_like(x, target_shape, dims_reduced):
        """
        Expand tensor x to match target_shape by adding dimensions at dims_reduced
        """
        if isinstance(dims_reduced, int):
            dims_reduced = (dims_reduced,)

        for dim in sorted(dims_reduced):
            x = cp.expand_dims(x, axis=dim)

        return cp.broadcast_to(x, target_shape)

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
    def _ensure_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    @staticmethod
    def _accumulate_grad(tensor, grad):
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad