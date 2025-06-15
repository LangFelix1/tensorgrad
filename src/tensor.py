import cupy as cp

class Tensor:
    def __init__(self, data, _prev=(), _op=''):
        if not isinstance(data, cp.ndarray):
            data = cp.array(data, dtype=cp.float32)
        else:
            data = data.astype(cp.float32)

        if data.ndim == 1:
            data = data[:, cp.newaxis]  # shape (n,) -> (n, 1)
    
        self.data = data
        self.grad = cp.zeros_like(data)

        self._backward = lambda: None
        self._prev = set(_prev)
        self._op = _op

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot add Tensor with object of type {type(other)}")

        if self.data.shape != other.data.shape:
            raise ValueError(f"Shape mismatch in addition, {self.data.shape} + {other.data.shape}")

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            if not isinstance(other, Tensor):
                raise TypeError(f"Cannot add Tensor with object of type {type(other)}")
            
            if self.data.ndim != 2 or other.data.ndim != 2:
                raise ValueError("Only 2D tensors supported for matrix multiplication")

            if self.data.shape[1] != other.data.shape[0]:
                raise ValueError(f"Shape mismatch for matrix multiplication, {self.data.shape} + {other.data.shape}")
            
            out_data = cp.matmul(self.data, other.data)
            out = Tensor(out_data, (self, other), '*')

            def _backward():
                self.grad += cp.matmul(out.grad, other.data.T)
                other.grad += cp.matmul(self.data.T, out.grad)
            out._backward = _backward

            return out
        
        elif isinstance(other, (float, int, cp.number)):
            out_data = self.data * other
            out = Tensor(out_data, (self,), f'*{other}')

            def _backward():
                self.grad += other * out.grad
            out._backward = _backward

            return out

        else:
            raise TypeError(f"Cannot multiply Tensor with type {type(other)}")
    
    def sum(self):
        out_data = cp.sum(self.data)
        out = Tensor(out_data, _prev=(self,), _op='sum')

        def _backward():
            self.grad += out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        if self.data.size != 1:
            raise RuntimeError("backward not supported for non-scalar output.")
        self.grad = cp.ones_like(self.data)

        # Simple topological traversal
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

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1
    
    def __rmul__(self, other):
        return self * other