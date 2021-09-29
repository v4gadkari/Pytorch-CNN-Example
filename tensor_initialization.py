#Tensor initialization


import torch 




#can also set tensor to gpu if you have cuda, else cpu
device = "cuda" if torch.cuda.is_available() else "cpu"


#creating a tensor
# 2 rows, 3 columns
my_tensor = torch.tensor([[1,2,3], [4, 5, 6]], dtype=torch.float32,
requires_grad=True)

#requires_grad -> argument needed when you want gradients to be 
#calculated for tensor

#information about tensor
print(my_tensor)
print(my_tensor.grad_fn)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)


#other common initializations

x = torch.empty(size=(3,3))

#un-initalized tensor

print(x)

#tensor with zeroes

x = torch.zeros(size=(3,3))

print(x)


#random tensor with values from a uniform dist

x = torch.rand(size=(3,3))

print(x)

#tensor with values of 1

x = torch.ones(size=(3,3))

print(x)
#identity matrix (1 across diagonals)

x = torch.eye(5,5)

print(x)


x = torch.arange(start=0, end=5, step=1)

print(x)


#print 10 values between 0.1 and 1
x = torch.linspace(start=0.1, end=1, steps=10)

print(x)


#initialized tensor of values from a normal dist
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)

print(x)

x = torch.empty(size=(1,5)).uniform_(0,1)

print(x)


#identity matrix of ones on diagonal (3 x 3)
x = torch.diag(torch.ones(3))

print(x)



#how to initialize and convert tensors to other types
#  (int, float, double)


tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short()) #int16
print(tensor.long())
print(tensor.half())
print(tensor.float())# used often
print(tensor.double())#float 64




#convert between numpy and torch
import numpy as np 

np_array = np.zeros((5,5))

torch_array = torch.from_numpy(np_array)

np_array_back = torch_array.numpy()

print(torch_array)
print(np_array_back)





