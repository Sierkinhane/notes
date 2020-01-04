import torch.nn as nn
class Fc(nn.Module):
    """docstring for fc"""
    def __init__(self, ):
        super(Fc, self).__init__()
        self.extract = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
        nn.Linear(8*8*32, 128),
        # nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        # nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2),
        # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.extract(x)
        x = x.view(-1, 8*8*32)
        # print(x.shape)
        x = self.fc(x)
        return x

