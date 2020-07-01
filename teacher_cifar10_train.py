import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import utils.app_constants as constants
from models.teacher_model import TeacherNetwork
import utils.common_utils
from torch.autograd import Variable as var
import logging
import datetime as datetime

use_cuda = utils.common_utils.torch.cuda.is_available()
device = utils.common_utils.torch.device("cuda:0" if use_cuda else "cpu")
utils.common_utils.torch.backends.cudnn.benchmark = True

losses_df = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss', 'Accuracy'])
logging.basicConfig(filename='./training_teacher.log', filemode='w', format='%(levelname)s - %(message)s')

'''
The cells use feature maps from every preceding output in the subsequent cells,
increasing the number of feature maps for every next cell by k
'''
k = constants.K
cell_connections_in = [k, 2 * k, 3 * k, 4 * k, 5 * k, 6 * k]
cell_connections_out = [k] * 6

'''
The stages also use feature maps from every preceding output in the subsequent 
cells, increasing the number of feature maps for every next stage linearly. 
This is due to the fact that the 1 x 1 convolution at the end of every stage 
reduces the output feature maps to size 'k'
'''

stage_connections_in = [3, k, 2 * k, 3 * k]
stage_connections_out = [k] * 4

teacher = TeacherNetwork(cell_connections_in,
                         cell_connections_out,
                         stage_connections_in,
                         stage_connections_out,
                         num_classes=constants.CIFAR_CLASSES).cuda()

'''
Defining the Loss function for use on Teacher module
As the nature of the problem is a classification prob
we use Cross Entropy Loss
'''
cec = utils.common_utils.nn.CrossEntropyLoss()

## Printing the number of parameters
s = sum(np.prod(list(p.size())) for p in teacher.parameters())
print('Number of parameters: ', s)

'''                 
Defining the Optimizer. We use ADAM with weight decay to optimize our model
'''
optimizer = optim.Adam(teacher.parameters(), lr=constants.LR, weight_decay=constants.WD)

'''Loading the CIFAR-10 Dataset'''
train_set, test_set = utils.common_utils.load_datasets(constants.CIFAR_10_DATASET)

'''
Main training loop
'''
for e in range(constants.EPOCHS):
    for i, (images, labels) in enumerate(train_set):
        images = var(images.cuda())
        labels = var(labels.cuda())
        optimizer.zero_grad()

        prediction = teacher(images)
        loss = cec(prediction, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % constants.DISPLAY_BATCH == 0:
            print('Epoch :', e + 1, 'Batch :', (i + 1), 'Loss :', float(loss.data))
            logging.INFO('Epoch %s || Batch %s || Loss %s',str(e+1),str(i+1),str(loss.data))
            losses_df.append({'Epoch': e + 1, 'Batch': i + 1, 'Loss': float(loss.data), 'Accuracy': -1.0})

    ## End of an epoch accuracy check
    accuracy = float(utils.common_utils.validate(teacher, test_set))
    print('Epoch :', e + 1, 'Batch :', (i + 1), 'Loss :', float(loss.data), ' Accuracy: ', accuracy, '%')
    logging.INFO("Epoch %s || Batch %s || Loss %s || Accuracy %s", str(e + 1), str(i + 1), str(loss.data),str(accuracy))
    losses_df.append({'Epoch': e + 1, 'Batch': i + 1, 'Loss': float(loss.data), 'Accuracy': accuracy})

    if (e + 1) % constants.SAVE_N == 0:
        ## Saving the model state every N epochs
        utils.common_utils.saveModel(e, teacher, optimizer, loss, constants.SAVE_PATH)

# save dataframe after complete execution
losses_df.to_csv(path_or_buf='./run_data.csv', sep='\t')
