import torch 


batch_size = 10
features = 25 

x = torch.rand((batch_size, 25))

print(x.shape)

print(x[0].shape)

print(x[:, 0].shape)


#get all 25 features for first batch 
print(x[0])


#get first feature across all batches 
print(x[:, 0])


#get third example, first 10 features 

print(x[2, 0:10].shape)

x[0, 0] = 100

# Fancy indexing 

#pick three elements that match the indices
x = torch.arange(10)

indices = [2, 5, 8]

print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1,0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)


#more advanced indexing 

x = torch.arange(10)

#pick elements less than 2 or greater than 8
print(x[(x < 2) | (x > 8)])

#do the same for and 

print(x[x.remainder(2) == 0])


#useful operations

#condition, true condition, false condition
print(torch.where(x > 5, x, x * 2))

print(torch.tensor([0,0,1,2,2, 3, 4]).unique())

print(x.ndimension()) # 5x5x5
print(x.numel())


