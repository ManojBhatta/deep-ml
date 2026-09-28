class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		other = Value(other) if not isinstance(other, Value) else other
		out = Value(self.data + other.data, (self, other), "+")

		def _backward():
			self.grad += out.grad
			other.grad += out.grad

		out._backward = _backward
		return  out

	def __mul__(self, other):
		other = Value(other) if not isinstance(other, Value) else other
		out = Value(self.data * other.data, (self, other), "+")

		def _backward():
			self.grad += out.grad * other.data
			other.grad += out.grad * self.data

		out._backward = _backward
		return  out

	def relu(self):
		out = Value(max(0, self.data), (self,), "relu")

		def _backward():
			g = 0 if self.data < 0 else 1
			self.grad += out.grad * g

		out._backward = _backward
		return  out

	def backward(self):
		topo = []
		visited = set()

		def build_topo(v:Value):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		
		build_topo(self)
		self.grad = 1
		for v in reversed(topo):
			v._backward()
		