import torch as torch
import torch.nn as nn

import utils.app_constants as constants


class DenseConnection(nn.Module):
    '''
    Defines the composite layer for the dense connections between cells and stages
    Each composite layer takes N features as input from the concatenated features
    of all previous layers and produces a constants.K feature output where 'constants.K' is the growth
    rate of the dense network.
    The network consists of :
    BN -> ReLU -> 1x1 ->[4k feature maps] BN -> ReLU -> 3x3 -> [constants.K feature maps]
    where 'constants.K' is the growth rate for the dense layer
    For this setup we have put constants.K = 32

    Refers to paper: https://arxiv.org/abs/1608.06993
    [Densely Connected Convolutional Networks, Huang et. al]
    '''

    def __init__(self, in_channels, out_channels):
        super(DenseConnection, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(4 * out_channels)
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=4 * constants.K, kernel_size=1, stride=1)
        self.cnn3 = nn.Conv2d(in_channels=4 * out_channels, out_channels=constants.K, kernel_size=3, stride=1)
        self.padder = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.activation_function(x)
        x = self.cnn1(x)

        x = self.batch_norm2(x)
        x = self.activation_function(x)
        x = self.cnn3(x)
        x = self.padder(x)

        return x;


class Cell(nn.Module):
    '''
    Defines an individual cell block in the densely connected
    teacher network.
    The network is split into two parallel convolutiosn each
    working on half the feature maps. The two convolutions use
    kernel sizes 1x1 and 3x3 respectively and concat before the
    next layer
    '''

    def __init__(self):
        super(Cell, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.cnn1 = nn.Conv2d(in_channels=int(constants.K / 2), out_channels=int(constants.K / 2), kernel_size=1,
                              stride=1)
        self.padder = nn.ReflectionPad2d(1)

    def forward(self, x):
        '''
        :param x: tensor input to the cell module
        :param constants.K: convolution batch size
        :return: 
        '''

        (path1, path2) = torch.split(x, split_size_or_sections=[int(constants.K / 2), int(constants.K / 2)], dim=1)
        path1 = self.cnn1(path1)
        path2 = self.cnn3(path2)

        x = torch.cat([path1, path2], 1)
        x = self.batch_norm(x)
        x = self.activation_function(x)
        return x


class Stage(nn.Module):
    def __init__(self, dense_connections_in, dense_connections_out):
        super(Stage, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(constants.K)
        self.cnn = nn.Conv2d(in_channels=constants.K, out_channels=constants.K, kernel_size=1, stride=1)
        self.cell = Cell()
        self.cell_dense_connections = nn.ModuleList(
            [DenseConnection(dense_connections_in[i], dense_connections_out[i]) for i in range(6)])

    def forward(self, x):

        cell_results = []
        s1 = self.cnn(x)
        s1 = self.batch_norm(s1)
        s1 = self.activation_function(s1)
        x = s1

        cell_results.append(x)  # initial input added to list
        for i in range(6):

            concatenated_tensor = torch.cat(cell_results, 1)
            # using dense connection compositing
            if i != 0:
                composited = self.cell_dense_connections[i - 1].forward(concatenated_tensor)

                thisCell = self.cell.forward(composited)
                cell_results.append(thisCell);

            else:
                thisCell = self.cell.forward(concatenated_tensor)
                cell_results.append(thisCell)

        concatenated_tensor = torch.cat(cell_results, 1)

        x = self.cell_dense_connections[5].forward(concatenated_tensor)

        x = self.cnn(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x


class TeacherNetwork(nn.Module):
    '''
    Teacher Network defined
    '''

    def __init__(self, cell_dense_connections_in, cell_dense_connections_out, stage_dense_connections_in,
                 stage_dense_connections_out):
        super(TeacherNetwork, self).__init__()

        self.initial_cnn = nn.Conv2d(3, constants.K, 1)
        self.batch_norm = nn.BatchNorm2d(constants.K)
        self.stage_dense_connections = nn.ModuleList(
            [DenseConnection(stage_dense_connections_in[i], stage_dense_connections_out[i]) for i in range(4)])

        self.stage = Stage(cell_dense_connections_in, cell_dense_connections_out)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.linear = nn.Linear(in_features=7200, out_features=43)

    def forward(self, x):
        stage_results = []

        '''
        Defining a transistion block for the network, as per DenseNet in order
        to take the input image and prepare it for the stage modules
        '''

        x = self.initial_cnn(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        stage_results.append(x)
        for i in range(4):
            concatenated_tensor = torch.cat(stage_results, 1)

            if i != 0:
                composited = self.stage_dense_connections[i - 1].forward(concatenated_tensor)

                thisStage = self.stage.forward(composited)
                stage_results.append(thisStage);

            else:
                thisStage = self.stage.forward(concatenated_tensor)
                stage_results.append(thisStage)

        concatenated_tensor = torch.cat(stage_results, 1)

        x = self.stage_dense_connections[3].forward(concatenated_tensor)

        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x;
