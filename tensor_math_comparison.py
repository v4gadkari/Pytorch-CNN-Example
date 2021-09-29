import torch 


x = torch.tensor([1,2,3])

y = torch.tensor([9, 8, 7])

#addition 

z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
print(z2)

z3 =  x + y
print(z3)


#subtraction

z4 = x - y

#division 

z5 = torch.true_divide(x, y)

#element wise division if they are same shape
#if y is an integer, divide each element by integer
print(z5)

print(torch.true_divide(x, 2))

#inplace operations

t = torch.zeros(3)
t.add_(x)
print(t)

#inplace operation - mutate tensor without creating a copy
#indicated by underscore



#exponentiation

z = x.pow(2)
#element wise exponents
print(z)

z = x ** 2 #same thing 


# simple comparisons

z = x > 0 
print(z) 


#matrix multiplacation

x1 = torch.rand((2,5))

x2 = torch.rand((5,3))




x3 = torch.mm(x1, x2) #2 x 3

x3 = x1.mm(x2)

print(x3)


#matrix exponentiation 

matrix_exp = torch.rand((5,5))
print(matrix_exp.matrix_power(3))


#element-wise multiplacation 

z = x * y
print(z)


#dot product 

dot_p = x.dot(y)
print(dot_p)

dot_p = torch.dot(x,y)
print(dot_p)


#batch matrix multiplacation 

batch = 5
n = 10
m = 20
p = 30 


tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1, tensor2)

print(out_bmm) # (batch, n, p), (batch, 10)



#examples of broadcasting 

x1 = torch.rand((5,5))
x2 = torch.rand((1, 5))

#x2 vector will be subtracted by each row of matrix by
#expanding to the dimensions of x1
z = x1-x2

print(z)


#other useful tensor operations

#sums over x
sum_x = torch.sum(x, dim=0)

#dim = 0 for vector
#dim = 1 for column
print(sum_x)

#returns the max value from the tensor
values, indices = torch.max(x, dim=0)

print(values, indices)

values, indices = torch.min(x, dim=0)

#abs value element wise
abs_max = torch.abs(x)

#same as torch.max except it returns the index
z = torch.argmax(x, dim=0)
print(z)
z = torch.argmin(x, dim=0)


mean_x = torch.mean(x.float(), dim=0)
print(mean_x)


tensor_rand = torch.tensor([[1,2,3], [4,5,6]])

#across columns
values, indices = torch.max(tensor_rand, dim=1)

#across rows
values_2, indices_2 = torch.max(tensor_rand, dim=0)


arg_max =torch.argmax(tensor_rand, dim=1)

arg_max_two = torch.argmax(tensor_rand, dim=0)


print([values, indices])
print(arg_max)


print([values_2, indices_2])
print(arg_max_two)


#element wise comparison of which are equal
z = torch.eq(x, y)

print(z)#all false

#sort in ascending order
print(torch.sort(y, dim=0, descending=False))

sorted_y, indices = torch.sort(y, dim=0, descending=False)

print([sorted_y, indices])



#all elements less than x are set to zero
z = torch.clamp(x, min=0)


x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)

#check for 1
z = torch.any(x)
print(z)

#checks for all to be 1
z = torch.all(x)
print(z)



