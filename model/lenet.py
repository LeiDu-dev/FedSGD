from abc import ABC

from torch import nn


class LeNet5(nn.Module, ABC):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()  # super用法:继承父类nn.Model的属性，并用父类的方法初始化这些属性

        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_dim, 6, 5, 1, 2),  # out_dim=6, kernel_size=5, stride=1, padding=2
            nn.Conv2d(in_dim, 6, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # kernel_size=2, padding=2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # in_features=400, out_features=120
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv = out_conv2.view(out_conv2.size(0), -1)

        out = self.fc(out_conv)
        return out


def lenet5():
    """ return a LeNet 5 object
    """
    return LeNet5(3, 10)
