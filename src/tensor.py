import cupy as cp

class Tensor:
    def __init__(self, data, _prev=(), _op=''):
        if not isinstance(data, cp.ndarray):
            data = cp.array(data, dtype=cp.float32)
        else:
            data = data.astype(cp.float32)
    
        self.data = data
        self.grad = cp.zeros_like(data)

        self._backward = lambda: None
        self._prev = set(_prev)
        self._op = _op

    def __add__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += Tensor.unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor.unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += Tensor.unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor.unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data ** other.data, (self, other), f'**{other}')

        def _backward():
            self.grad += Tensor.unbroadcast((other.data * self.data**(other.data - 1)) * out.grad, self.data.shape)
            other.grad += Tensor.unbroadcast((out.data * cp.log(self.data)) * out.grad, other.data.shape)
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = Tensor._ensure_tensor(other)       
        out_data = cp.matmul(self.data, other.data)
        out = Tensor(out_data, (self, other), '@')

        def _backward():
            self.grad += Tensor.unbroadcast(cp.matmul(out.grad, other.data.T), self.data.shape)
            other.grad += Tensor.unbroadcast(cp.matmul(self.data.T, out.grad), other.data.shape)
        out._backward = _backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __rmul__(self, other):
        return self * other

    def sum(self, dim=None, keepdim=False):
        out_data = cp.sum(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(out_data, _prev=(self,), _op='sum')

        def _backward():
            grad = out.grad
            if not keepdim and dim is not None:
                grad = Tensor.expand_like(grad, self.data.shape, dim)
            self.grad += grad
        out._backward = _backward

        return out
    
    def backward(self):
        self.grad = cp.ones_like(self.data)

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
            t._backward()
    
    @staticmethod
    def expand_like(x, target_shape, dims_reduced):
        """
        Expand tensor x to match target_shape by adding dimensions at dims_reduced
        """
        if isinstance(dims_reduced, int):
            dims_reduced = (dims_reduced,)

        for dim in sorted(dims_reduced):
            x = cp.expand_dims(x, axis=dim)

        return cp.broadcast_to(x, target_shape)

    @staticmethod
    def unbroadcast(x, target_shape):
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