"""
Lab 5: Frameworks and automatic differentiation
Course: Machine Learning [CII5SE039CI], Group 03
Institution: Faculty of Computer Science, Graphics and Architecture, Vistula University
Author: Emir Shadybekov (Album: 73429)

Task 1, 2, 3: Scalar Automatic Differentiation Engine (Value node)
- Stores data, gradient, parent nodes, and local derivative rule.
- Implements addition, multiplication, powers, and ReLU.
- Implements topological backward pass with gradient accumulation (+=).
"""

import math
from typing import Set, Tuple, Union


class Value:
    """
    A scalar Value node representing a computational graph vertex.
    Stores numerical data, accumulated gradient, parent nodes, operation symbol,
    and a local backward propagation rule.
    """
    def __init__(self, data: Union[int, float], _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: Set['Value'] = set(_children)
        self._op: str = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    # --- Task 2: Core Arithmetic Operations with Graph Construction ---

    def __add__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Local derivatives: d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other: Union['Value', int, float]) -> 'Value':
        return self + other

    def __mul__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Local derivatives: d(out)/d(self) = other, d(out)/d(other) = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other: Union['Value', int, float]) -> 'Value':
        return self * other

    def __pow__(self, other: Union[int, float]) -> 'Value':
        assert isinstance(other, (int, float)), f"Only int/float powers supported, got {type(other)}"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n - 1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def relu(self) -> 'Value':
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            # ReLU derivative: 1.0 if x > 0 else 0.0
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad
        out._backward = _backward

        return out

    # --- Auxiliary Operations for Training & Loss Computation ---

    def __neg__(self) -> 'Value':
        return self * -1.0

    def __sub__(self, other: Union['Value', int, float]) -> 'Value':
        return self + (-other)

    def __rsub__(self, other: Union['Value', int, float]) -> 'Value':
        return Value(other) + (-self)

    def __truediv__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union['Value', int, float]) -> 'Value':
        return Value(other) * (self ** -1.0)

    def log(self, eps: float = 1e-15) -> 'Value':
        """Natural logarithm with epsilon clipping for numerical stability."""
        val = max(self.data, eps)
        out = Value(math.log(val), (self,), 'log')

        def _backward():
            self.grad += (1.0 / val) * out.grad
        out._backward = _backward

        return out

    def exp(self) -> 'Value':
        """Exponential function."""
        # Clip to prevent overflow
        val = min(max(self.data, -500.0), 500.0)
        e = math.exp(val)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward

        return out

    def sigmoid(self) -> 'Value':
        """Numerically stable Sigmoid activation."""
        if self.data >= 0:
            z = math.exp(-self.data)
            s = 1.0 / (1.0 + z)
        else:
            z = math.exp(self.data)
            s = z / (1.0 + z)
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # Derivative of sigmoid: s * (1 - s)
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward

        return out

    # --- Task 3: Backward Pass via Topological Sort ---

    def backward(self):
        """
        Executes reverse-mode automatic differentiation:
        1. Topologically sorts all ancestor nodes in the computation graph.
        2. Initializes the root gradient to 1.0 (d(root)/d(root) = 1).
        3. Visits nodes in reverse topological order.
        4. Accumulates incoming gradients (+=) to preserve multivariable chain rule branching.
        """
        topo = []
        visited = set()

        def build_topo(v: 'Value'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Base gradient for the output node
        self.grad = 1.0

        # Propagate gradients backwards
        for node in reversed(topo):
            node._backward()
