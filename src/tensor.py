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
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
    
    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other), f'**{other}')

        def _backward():
            self.grad += (other.data * self.data**(other.data - 1)) * out.grad
            other.grad += (out.data * cp.log(self.data)) * out.grad
        out._backward = _backward
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)        
        out_data = cp.matmul(self.data, other.data)
        out = Tensor(out_data, (self, other), '@')

        def _backward():
            self.grad += cp.matmul(out.grad, other.data.T)
            other.grad += cp.matmul(self.data.T, out.grad)
        out._backward = _backward

        return out

    def sum(self):
        out_data = cp.sum(self.data)
        out = Tensor(out_data, _prev=(self,), _op='sum')

        def _backward():
            self.grad += out.grad
        out._backward = _backward

        return out
    
    def backward(self):
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
    
    def __rmul__(self, other):
        return self * other