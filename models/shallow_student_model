class StudentNetwork(nn.Module):
  def __init__(self):
    super(StudentNetwork,self).__init__()
    self.activation_function = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm2d(32)
    self.batch_norm3 = nn.BatchNorm2d(64)
    self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.cnn1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.cnn2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.cnn3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
    self.fc = nn.Linear(in_features=2048,out_features=43)

  def forward(self,x):
    x = self.cnn1(x)
    x = self.batch_norm1(x)
    x = self.activation_function(x)

    x = self.cnn2(x)
    x = self.batch_norm3(x)
    x = self.activation_function(x)

    x = self.max_pool(x)

    x = self.cnn3(x)
    x = self.batch_norm1(x)
    x = self.activation_function(x)

    x = self.max_pool(x)

    x = x.view(x.size(0),-1)
    x = self.fc(x)

    return x;