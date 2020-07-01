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
    super(StudentNetwork,self).__init__()
    self.activation_function = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm2d(32)
    self.batch_norm2 = nn.BatchNorm2d(32)
    self.batch_norm3 = nn.BatchNorm2d(64)
    self.batch_norm4 = nn.BatchNorm2d(64)
    self.batch_norm5 = nn.BatchNorm2d(128)
    self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
    self.cnn1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.cnn2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.cnn3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.cnn4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.cnn5 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.fc = nn.Linear(in_features=2048,out_features=43)

  def forward(self,x):
    x = self.cnn1(x)
    x = self.batch_norm1(x)
    x = self.activation_function(x)
    x = self.cnn2(x)
    x = self.batch_norm2(x)
    x = self.activation_function(x)
    x = self.max_pool(x)
    x = self.cnn3(x)
    x = self.batch_norm3(x)
    x = self.activation_function(x)
    x = self.cnn4(x)
    x = self.batch_norm4(x)
    x = self.activation_function(x)
    x = self.max_pool(x)
    x = self.cnn5(x)
    x = self.batch_norm5(x)
    x = self.activation_function(x)
    x = self.avg_pool(x)
    x = x.view(x.size(0),-1)
    x = self.fc(x)

    return x;
