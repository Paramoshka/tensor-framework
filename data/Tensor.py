import numpy as np


class Tensor(object):
    def __init__(self,
                 data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 tensor_id=None):

        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = dict()

        if tensor_id is None:
            tensor_id = np.random.randint(0, 100000)
        self.tensor_id = tensor_id
        if creators is not None:
            for c in creators:
                c: Tensor
                if c.tensor_id not in c.children:
                    c.children[self.tensor_id] = 1
                else:
                    c.children[self.tensor_id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False

        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                grad_origin: Tensor
                if self.children[grad_origin.tensor_id] == 0:
                    raise Exception("Cannot backward grad of tensor with no children")
                else:
                    self.children[grad_origin.tensor_id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is not None):
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        return Tensor(self.data + other.data)

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
