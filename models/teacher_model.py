import torch as torch
import torch.nn as nn

import utils.app_constants as constants


class Cell(nn.Module):
    '''
    Defines an individual cell block in the densely connected
    teacher network.
    The network is split into two parallel convolutiosn each
    working on half the feature maps. The two convolutions use
    kernel sizes 1x1 and 3x3 respectively and concat before the
    next layer
    '''

    def __init__(self, cell_in_channels, cell_out_channels):
        super(Cell, self).__init__()

        self.activation_function = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(cell_out_channels)

        ## Reflect padding is used for the 3 x 3 convolution as it creates a
        ## feature map of size 30 X 30, and needs to be padded to 32 x 32
        ## in order to concatenate with the 1 x 1 conv tensor

        self.cnn3 = nn.Conv2d(in_channels=int(cell_in_channels / 2),
                              out_channels=int(cell_out_channels / 2),
                              kernel_size=3, padding=1, padding_mode='reflect',
                              stride=1)

        self.cnn1 = nn.Conv2d(in_channels=int(cell_in_channels / 2),
                              out_channels=int(cell_out_channels / 2),
                              kernel_size=1, stride=1)

        self.split_size = int(cell_in_channels / 2)

    def forward(self, x):
        (path1, path2) = torch.split(x, split_size_or_sections=[self.split_size, self.split_size], dim=1)
        path1 = self.cnn1(path1)

        path2 = self.cnn3(path2)

        x = torch.cat([path1, path2], 1)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x


class Stage(nn.Module):
    '''
    *Six cells are used to establish the direct connection between different
    layers, making full use of the feature maps of each layer*

    The outputs from each of the cells, as well as the 1 x 1 convolution are
    accumulated into the input for the next cell

    The two 1 x 1 convolutions are used to reduce the number of feature maps
    when connecting between the two stages
    '''

    def __init__(self, cell_connections_in, cell_connections_out, stage_in, stage_out):
        super(Stage, self).__init__()

        self.activation_function = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(k)

        self.cnn1 = nn.Conv2d(in_channels=stage_in,
                              out_channels=k, kernel_size=1,
                              stride=1)

        self.cnn2 = nn.Conv2d(in_channels=7 * k,
                              out_channels=stage_out, kernel_size=1,
                              stride=1)

        ## Densely connected six cell blocks
        self.cells = nn.ModuleList([
            Cell(cell_connections_in[i],
                 cell_connections_out[i]) for i in range(6)
        ])

    def forward(self, x):
        cell_results = []
        x = self.cnn1(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        cell_results.append(x)
        for i in range(6):
            x = torch.cat(cell_results, 1)
            x = self.cells[i](x)
            cell_results.append(x);

        x = torch.cat(cell_results, 1)

        x = self.cnn2(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x


class TeacherNetwork(nn.Module):
    '''
    Finally, we define the teacher network which consists of 4 stage modules connected
    in a dense fashion, with each stage producing a 'k' feature maps where 'k' is the
    growth rate of the network.
    Stage 0 takes the input tensor which has 3 x H X W tensor and outputs a k x H x W
    tensor. The remaining Stages take 'k' feature maps as input and output 'k' feature
    maps

    Finally, the Stage 3 output is pooled using a 3 x 3 max pooling with stride of 2
    and finally a fully connected linear layer which produces the probability vector
    for classification.
    '''

    def __init__(self, cell_onnections_in, cell_connections_out, stage_connections_in, stage_connections_out):
        super(TeacherNetwork, self).__init__()

        self.stages = nn.ModuleList([Stage(cell_onnections_in,
                                           cell_connections_out,
                                           stage_connections_in[i],
                                           stage_connections_out[i])
                                     for i in range(4)])

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation_function = nn.ReLU()
        self.linear = nn.Linear(in_features=131072, out_features=constants.CIFAR_CLASSES)

    def forward(self, x):
        stage_results = []
        for i in range(4):
            if i != 0:
                x = torch.cat(stage_results, 1)
                x = self.stages[i](x)
                stage_results.append(x);

            else:
                x = self.stages[0](x)
                stage_results.append(x)

        x = torch.cat(stage_results, 1)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x;
