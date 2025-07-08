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