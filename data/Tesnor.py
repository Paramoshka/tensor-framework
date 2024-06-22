import numpy as np


class Tesnor(object):
    def __init__(self, data, creators=None, creation_op=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None

    def backward(self, grad):
        self.grad = grad

        if self.creation_op == "add":
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)

    def __add__(self, other):
        return Tesnor(self.data + other.data, creators=[self, other], creation_op="add")

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

# sample
# x = Tesnor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = Tesnor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
#
# z = x + y
# z.backward(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
