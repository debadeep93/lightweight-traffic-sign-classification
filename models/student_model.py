from torch import nn
import utils.app_constants as constants
from models.teacher_model import Stage


class StudentNetwork(nn.Module):
    '''
    Simple CNN Model containing 5 Conv Layers
    2 Max Pool Layers, 1 Avg Pool Layer
    and 1 FC Layer
    '''
    def __init__(self):
        super(Stage,self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(constants.K)
        self.padding = nn.ReflectionPad2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=2)
        self.cnn1 = nn.Conv2d(in_channels=3,out_channels=constants.K,kernel_size=3,stride=1)
        self.cnn2_5 = nn.Conv2d(in_channels=constants.K,out_channels=constants.K,kernel_size=3,stride=1)
        self.fc = nn.Linear(in_features=4096,out_features=43)

        def forward(self,x):
            x = self.cnn1(x)
            x = self.batch_norm(x)
            x = self.activation_function(x)
            x = self.cnn2_5(x)
            x = self.batch_norm(x)
            x = self.activation_function(x)
            x = self.max_pool(x)
            x = self.cnn2_5(x)
            x = self.batch_norm(x)
            x = self.activation_function(x)
            x = self.cnn2_5(x)
            x = self.batch_norm(x)
            x = self.activation_function(x)
            x = self.max_pool(x)
            x = self.cnn2_5(x)
            x = self.batch_norm(x)
            x = self.activation_function(x)
            x = self.avg_pool(x)
            x = self.Linear(x)

            return x;