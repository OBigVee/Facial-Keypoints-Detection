## TODO: define the convolutional neural network architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
     
        #input image = 224x224  kernel size = 3 
        #CONV1
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (32, 222, 222)
        # after one pool layer, this becomes (32, 111, 111)
        self.conv1 = nn.Conv2d(1, 32, 3) #(222)
        self.pool = nn.MaxPool2d(2, 2) #(111)
        
        #CONV2
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = 109
        # the output Tensor for one image, will have the dimensions: (64, 109, 109)
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3) #(109)
        self.pool = nn.MaxPool2d(2, 2)
        
        #CONV3
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for one image, will have the dimensions: (128, 52, 52)
        # after one pool layer, this becomes (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)#(52)
        self.pool = nn.MaxPool2d(2, 2)
        
        #CONV4
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output Tensor for one image, will have the dimensions: (256, 24, 24)
        # after one pool layer, this becomes (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 3) #(24)
        self.pool = nn.MaxPool2d(2, 2)
        
        #CONV5
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output Tensor for one image, will have the dimensions: (512, 10, 10)
        # after one pool layer, this becomes (512, 5, 5)
        self.conv5 = nn.Conv2d(256, 512, 3) #(10)
        self.pool = nn.MaxPool2d(2, 2)
        
        #CONV6
        ## output size = (W-F)/S +1 = (5-3)/1 +1 = 3
        # the output Tensor for one image, will have the dimensions: (1024, 3, 3)
        # after one pool layer, this becomes (1024, 1, 1)
        self.conv6 = nn.Conv2d(512 ,1024, 3)#(3)
        self.pool = nn.MaxPool2d(2, 2)
        
        
        # linear layers
        self.fc1 = nn.Linear(1024 * 1 * 1, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.dropout = nn.Dropout(0.4)
        self.fc3 = nn.Linear(500, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
       
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        #FLATTEN THE IMAGES AS VECTOR FOR FULLY CONECTED LAYERS
        x = x.view(x.size(0),-1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x