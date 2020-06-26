import torch
from torch import nn, optim
import numpy as np

import utils.app_constants as constants
from models.teacher_model import TeacherNetwork
from utils.common_utils import *
from torch.autograd import Variable as var

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

'''
The cells use feature maps from every preceding output in the subsequent cells,
increasing the number of feature maps for every next cell by k
'''
k = constants.K
cell_connections_in = [k,2*k,3*k,4*k,5*k,6*k]
cell_connections_out = [k] * 6

'''
The stages also use feature maps from every preceding output in the subsequent 
cells, increasing the number of feature maps for every next stage linearly. 
This is due to the fact that the 1 x 1 convolution at the end of every stage 
reduces the output feature maps to size 'k'
'''

stage_connections_in = [3,k,2*k,3*k]
stage_connections_out = [k] * 4

teacher = TeacherNetwork(cell_connections_in,
                         cell_connections_out,
                         stage_connections_in,
                         stage_connections_out,
                         num_classes=constants.CIFAR_CLASSES)\
    .to(device)

'''
Defining the Loss function for use on Teacher module
As the nature of the problem is a classification prob
we use Cross Entropy Loss
'''
cec = nn.CrossEntropyLoss()

## Printing the number of parameters
s  = sum(np.prod(list(p.size())) for p in teacher.parameters())
print('Number of parameters: ',s)

'''
Defining the Optimizer. We use ADAM with weight decay to optimize our model
'''
optimizer = optim.Adam(teacher.parameters(),lr=constants.LR,weight_decay=constants.WD)

'''Loading the CIFAR-10 Dataset'''
train_set,test_set = load_datasets(constants.CIFAR_10_DATASET)

'''
Main training loop
'''
for e in range(constants.EPOCHS):
    for i,(images,labels) in enumerate(train_set):
        images = var(images.to(device))
        labels = var(labels.to(device))
        optimizer.zero_grad()

        prediction = teacher(images)
        loss = cec(prediction,labels)
        loss.backward()
        optimizer.step()

        if (i+1) % constants.DISPLAY_BATCH == 0:
            print('Epoch :',e+1,'Batch :',(i+1),'Loss :',float(loss.data))

    ## End of an epoch accuracy check
    accuracy = float(validate(teacher,test_set))
    print('Epoch :',e+1,'Batch :',(i+1),'Loss :',float(loss.data),' Accuracy: ',accuracy,'%')

    if (e+1) % constants.SAVE_N == 0:
        ## Saving the model state every N epochs
        saveModel(e,teacher,optimizer,loss,constants.SAVE_PATH)

