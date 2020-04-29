import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool3d(x, x.shape[2:])

class _3d_cnn(nn.Module):
    def __init__(self, input_shape, output_dim):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(_3d_cnn, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(input_shape[0],  16, (5, 1, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(16, 16, (1, 9, 3), stride=(1, 2, 1)),
            nn.PReLU(),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1)),

            nn.Conv3d(16, 32, kernel_size=(4, 1, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=(1, 8, 3), stride=(1, 2, 1)),
            nn.PReLU(),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1)),
            
            nn.AvgPool3d(2)
        )

        # Compute number of input features for the last fully-connected layer
        input_shape = (1,) + input_shape
        x = Variable(torch.rand(input_shape), requires_grad=False)
        x = self.features(x)
        x = Flatten()(x)
        self.n = x.size()[1]

        self.fc1 = nn.Linear(self.n, num_features)
        self.fc2 = nn.Linear(num_features, output_dim)
        self.bn = nn.BatchNorm1d(self.n)
        self.bn2 = nn.BatchNorm1d(num_features)

    def forward(self, x):
        x = self.features(x)
        x = Flatten()(x)
        x = self.bn(x)
        x = self.bn2(F.relu(self.fc1(x)))
        x = F.sigmoid(self.fc2(x))
        return x