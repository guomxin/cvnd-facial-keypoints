## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    '''
    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W-F)/S+1 = (224-5)/1+1=220, the output tensor for one image will be (32, 220, 220)
        # after one pooling layer, this becomes (32, 110, 110) 
        self.conv1 = nn.Conv2d(1, 32, 5)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.4
        self.conv1_drop = nn.Dropout(p=0.1)
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 kernel
        # output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # dropout with p=0.4
        self.conv2_drop = nn.Dropout(p=0.2)
        
        # (64, 58, 58) => 1000
        self.fc1 = nn.Linear(64*53*53, 1000)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.3)
        
        # finally, create 136 output values, 2 for each of the 68 keypoint (x,y) pairs
        self.fc2 = nn.Linear(1000, 136)
    '''
    
    '''
    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 16  output channels/feature maps, 5x5 square convolution kernel
        # output size = (W-F)/S+1 = (224-5)/1+1=220, the output tensor for one image will be (16, 220, 220)
        # after one pooling layer, this becomes (16, 110, 110) 
        self.conv1 = nn.Conv2d(1, 16, 5)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.4
        self.conv1_drop = nn.Dropout(p=0.1)
        
        # second conv layer: 16 inputs, 32 outputs, 5x5 kernel
        # output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (32, 106, 106)
        # after another pool layer this becomes (32, 53, 53)
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        # dropout with p=0.4
        self.conv2_drop = nn.Dropout(p=0.2)
        
        # (32, 53, 53) => 1000
        self.fc1 = nn.Linear(32*53*53, 1000)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.3)
        
        # finally, create 136 output values, 2 for each of the 68 keypoint (x,y) pairs
        self.fc2 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # final output
        return x
    '''
    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32  output channels/feature maps, 4x4 square convolution kernel
        # output size = (W-F)/S+1 = (224-4)/1+1=221, the output tensor for one image will be (32, 221, 221)
        # after one pooling layer, this becomes (32, 110, 110) 
        self.conv1 = nn.Conv2d(1, 32, 4)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.1
        self.conv1_drop = nn.Dropout(p=0.1)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 kernel
        # output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        # after another pool layer this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # dropout with p=0.2
        self.conv2_drop = nn.Dropout(p=0.2)
        
        # third conv layer: 64 inputs, 128 outputs, 2x2 kernel
        # output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output tensor will have dimensions: (128, 53, 53)
        # after another pool layer this becomes (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 2)
        
        # dropout with p=0.3
        self.conv3_drop = nn.Dropout(p=0.3)

        # fourth conv layer: 128 inputs, 256 outputs, 1x1 kernel
        # output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output tensor will have dimensions: (256, 26, 26)
        # after another pool layer this becomes (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # dropout with p=0.4
        self.conv4_drop = nn.Dropout(p=0.4)
        
        # (256, 13, 13) => 1000
        self.fc1 = nn.Linear(256*13*13, 1000)
        
        # dropout with p=0.5
        self.fc1_drop = nn.Dropout(p=0.5)
        
        # 1000 => 1000
        self.fc2 = nn.Linear(1000, 1000)
        
        # dropout with p=0.6
        self.fc2_drop = nn.Dropout(p=0.6)
        
        # finally, create 136 output values, 2 for each of the 68 keypoint (x,y) pairs
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # four conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv4_drop(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # final output
        return x
