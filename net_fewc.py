import torch

from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class CNN(torch.nn.Module):
    def __init__(self,N_class=3,lamda=0):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.max_pool1=torch.nn.MaxPool1d(kernel_size=4,stride=2)
        self.conv2 = torch.nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.max_pool2 = torch.nn.MaxPool1d(kernel_size=4, stride=2)
        # KDD
        self.fn1 = torch.nn.Linear(256,128)
        self.fn2 = torch.nn.Linear(128,N_class)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        ### EWC
        self.lamda = lamda
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0),-1)
        x = self.fn1(x)
        x = self.relu(x)
        av = self.fn2(x)
        x = self.sigmoid(av)
        return x


class CNN2D(torch.nn.Module):
    """docstring for CNN_NORMAL"""
    def __init__(self, N_class=5, lamda=0):
        super(CNN2D, self).__init__()
        self.avg_kernel_size = 4
        self.i_size = 16
        self.num_class = N_class
        self.input_space = None
        self.input_size = (self.i_size, self.i_size, 1)
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 8*8*16
        )
        self.conv2 = torch.nn.Sequential(
            # torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*128
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*128
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 4*4*128
        )
        self.fc = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(4 * 4 * 128),
            # torch.nn.Linear(4 * 4 * 128, self.num_class, bias=True)

            # torch.nn.BatchNorm1d(4 * 4 * 32),
            torch.nn.Linear(4 * 4 * 32, self.num_class, bias=True)
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.lamda = lamda

    def features(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        # x = self.conv3(x)
        return x

    def logits(self, input_data):
        # x = self.avg_pool(input_data)
        # x = x.view(x.size(0), -1)
        x = input_data.view(input_data.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self,input_data):
        x = self.features(input_data)
        x = self.logits(x)
        x = self.sigmoid(x)
        return x


class CNNMnist(torch.nn.Module):
    def __init__(self,lamda=0):
        super(CNNMnist, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.lamda = lamda

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(torch.nn.Module):
    def __init__(self,lamda=0):
        super(CNNCifar, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.lamda = lamda

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNFashion_Mnist(torch.nn.Module):
    def __init__(self,lamda=0):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            # torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.fc = torch.nn.Linear(7*7*32, 10)
        self.lamda = lamda

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out