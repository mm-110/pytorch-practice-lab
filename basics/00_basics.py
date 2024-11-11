import torch
import numpy as np

# shapes
empty = torch.empty(1) # scalar
empty = torch.empty(3) # vector
empty = torch.empty(2, 3) # matrix
empty = torch.empty(2, 2, 3) # tensor, 3D
empty = torch.empty(2, 2, 2, 3) # tensor, 4D

print(f'use .shape attribute: {empty.shape}')
print(f'or .size() method: {empty.size()}')

# random numbers in range [0, 1]
rand = torch.rand(2, 2)
print(f'random: {rand}')
print(f'print shape of specific dimension: {rand.shape[0]}')

# zeros
zeros = torch.zeros(2, 2)
print(f'zeros: {zeros}')

# data type
print(f'data type: {zeros.dtype}')
zeros = torch.zeros(2, 2, dtype=torch.int) # int32
zeros = torch.zeros(2, 2, dtype=torch.float) # float32
zeros = torch.zeros(2, 2, dtype=torch.float16) # float16

# tensor
tensor = torch.tensor([5.5, 3])
print(f'tensor: {tensor}, data type: {tensor.dtype}')

# tensor requires_grad
# this will tell pytorch that it will need to calculate the gradients for this tensor
tensor = torch.tensor([5.5, 3], requires_grad=True)

# operations
# addition
x = torch.ones(2, 2)
y = torch.rand(2, 2)
z = x + y
z = torch.add(x, y)
print(f'addition: {z}')
# or y.add_(x) will add x to y in place

# subtraction
z = x - y 
z = torch.sub(x, y) # or y.sub_(x)

# multiplication
z = x * y 
z = torch.mul(x, y) # or y.mul_(x)

# division
z = x / y
z = torch.div(x, y) # or y.div_(x)

# slicing
x = torch.rand(5, 3)
print(f'slicing: {x[:, 0]}') # all rows, first column
print(f'slicing: {x[1, :]}') # return a tensor with second row, all columns
print(f'slicing: {x[1, 1]}') # return a tensor with the item at second row, second column
print(f'slicing .item(): {x[1, 1].item()}') # return the value at second row, second column

# reshaping
x = torch.rand(4, 4)
y = x.view(16) # reshape to 1D tensor
z = x.view(-1, 8) # -1 is inferred from other dimensions

# convert to numpy
a = torch.ones(5)
b = a.numpy() # a.clone().numpy() -> this will create a copy of the tensor
print(f'convert to numpy: {b}')
print(f'convert to numpy, type: {type(b)}')
a.add_(1)
print(a)
print(b) # also the other varaiable is updated

# convert from numpy
a = np.ones(5)
b = torch.from_numpy(a) # this shares the memory with the numpy array
c = torch.tensor(a)
print(f'convert from numpy: {b}')

a += 1
print(a)
print(b) # also the other varaiable is updated
print(c) # this variable is not updated

# move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand(2, 2).to(device) # move to GPU
x = torch.rand(2, 2, device=device) # create directly on GPU

# requires_grad = True -> tracks all operations on the tensor
print('\nGrad:')
x = torch.randn(3, requires_grad=True)
y = x + 2
# y was created as a result of an operation, so it has a grad_fn
# grad_fn shows the function that generated this variable
print(x)
print(y)
print(y.grad_fn)

z = y * y * 3
print(z)
z = z.mean()
print(z)

# compute gradients and backpropagation
print('\nBackpropagation:')
print(f'x: {x}')
print(f'grad: {x.grad}') # None
z.backward()
print(f'grad: {x.grad}') # dz/dx






