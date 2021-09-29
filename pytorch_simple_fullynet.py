#imports 
import torch 
import torch.nn as nn #neural network modules: layers, loss functions etc
import torch.optim as optim #optimization algorithms
import torch.nn.functional as F #functions with no parameters: activation functions
from torch.utils.data import DataLoader #dataset management
import torchvision.datasets as datasets #standard built in datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor #transformations on dataset 


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Create fully connected network

class NN(nn.Module):
    def __init__(self, input_size, num_classes): #MNIST: (28x28 Images) = 784 inputs
        super(NN, self).__init__()
        #super - calls the initialization method of parent class (nn.module)
        self.fc1 = nn.Linear(input_size, 50) #50 hidden layer nodes
        self.fc2 = nn.Linear(50, num_classes) #previous hidden layer is next input


    #forward pass function

    def forward(self, x):
        #perform layers from init method

        #pass x through fully connected layer
        #then pass it through relu activation
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x




#Create a CNN
class CNN(nn.Module):   #black and white = 1 , rgb = 3
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride = (1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride = (1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) #flatten
        x = self.fc1(x)

        return x



def train_cnn(train_loader, model, epochs):

    for epoch in range(epochs):
        for  (data, targets) in train_loader:

            data = data.to(device)
            targets = targets.to(device)


            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()






    



model_cnn = CNN().to(device)




    









#check if gpu is available, if not run on cpu

#hyperparameters

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size=64
num_epochs=1



#load data 


train_dataset = datasets.MNIST(
    root='dataset/',  #root - where is should save dataset
    train=True, #train set
    transform=transforms.ToTensor(), #when data is loaded it is a numpy array, this converts to pytorch tensors
    download=True #download if we dont have it in folder
    ) 


train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True) #shuffles each batch so we dont have same images being trained in each epoch


test_dataset = datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)


test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


#initialize network

#model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=learning_rate)
#the optimizer will update and tweak the hyperparameters


train_cnn(train_loader=train_loader, model=model_cnn, epochs=5)


#train network
'''

#epoch - iteration over a dataset
for epoch in range(num_epochs):
    #iterate over batches (data, targets)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device = device)


        #shape: (64, 1, 28, 28)
        # 64 - no. of images
        # 1 - number of channels (b&w)
        # 28x28 height and width

        #print([batch_idx, (data, targets)])
        #we want a long vector instead of matrix
        data = data.reshape((data.shape[0], -1))




        #forward 

        scores = model(data)
        loss = criterion(scores, targets)


        #backward
        #reset the gradients for each batch
        optimizer.zero_grad()

        #collect gradients of loss function
        #wrt each parameter
        loss.backward()

        #complete a gradient descent step
        optimizer.step()

'''







def check_cnn_accuracy(loader, model):
    num_correct = 0
    num_samples = 0 

    model.eval()


    with torch.no_grad():
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")


        for x,y in loader:

            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            


        print((float(num_correct)/float(num_samples))*100)


    model.train()



            

        

'''
#check accuracy on training & test

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    #let model know this is evaluation mode
    model.eval()

    #when we check accuracy we dont need to calculate
    #gradients
    with torch.no_grad():
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
        for x,y in loader:
            #x = x.to(device=device)
            #y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            
            #scores shape (64 x 10)
            #what is the max of those 10 digits?

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            #64 samples
            num_samples += predictions.size(0)

        print((float(num_correct)/float(num_samples)) * 100)
        #print(f'Got (num_correct)/(num_samples) with accuracy (float(num_correct)/float(num_samples)*100)')

    model.train()



check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
'''


check_cnn_accuracy(loader=train_loader, model=model_cnn)
check_cnn_accuracy(loader=test_loader, model=model_cnn)
    






#check if it gives correct shape

#model = NN(784, 10)

#x = torch.randn(64, 784) #64 minibatches x 784 image examples

#print(model(x).shape)
#we want the shape to be 64 x 10 (10 outputs for each batch)
