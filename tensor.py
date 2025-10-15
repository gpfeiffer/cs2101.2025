class Tensor:
    """a blueprint class for vectors and matrices"""

    # constructor
    def __init__(self, data):
        self.data = data

    # string representation
    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    # list like behaviour
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __eq__(self, other):
        return self.data == other.data

    # number like behavior
    def __add__(self, other):
        assert len(self) == len(other), "length mismatch"
        return type(self)([x + y for x, y in zip(self, other)])

    def __rmul__(self, other):
        return type(self)([other * x for x in self])

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + -other

class Vector(Tensor):
    """a class for vectors, adding and scaling"""

    def dot(self, other):  # v . v
        assert len(self) == len(other), "length mismatch"
        return sum(x * y for x, y in zip(self, other))

    def __matmul__(self, other):  # v @ m
        return Vector([self.dot(x) for x in other.T])

class Matrix(Tensor):
    """a class for matrices and matrix multiplication"""

    @property
    def T(self):
        return Matrix([Vector(x) for x in zip(*self)])

    def __matmul__(self, other):  # m @ m
        return Matrix([x @ other for x in self])

v = Vector([1, 2, 3])
m = Matrix([v, v, v])

ma = Matrix([
    Vector([1, 0, 1]),
    Vector([2, 1, 1]),
    Vector([0, 1, 1]),
    Vector([1, 1, 2])
])

mb = Matrix([
    Vector([1, 2, 1]),
    Vector([2, 3, 1]),
    Vector([4, 2, 2])
])
