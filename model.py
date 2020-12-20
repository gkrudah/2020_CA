import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=1)
        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self,x):
        # conv = self.conv1(F.relu(self.bn1(x)))
        # conv = self.conv2(F.relu(self.bn2(conv)))
        conv = self.conv1(self.swish(self.bn1(x)))
        conv = self.conv2(self.swish(self.bn2(conv)))
        shortcut = self.shortcut(x)
        #print(conv.shape, shortcut.shape)

        x = conv + shortcut

        return x


class ClassficationModel(nn.Module):
    def __init__(self, classes):
        super(ClassficationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.res1 = ResBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.res2 = ResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)

        #self.res3 = ResBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        #self.res4 = ResBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1)

        self.fc = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)

        #x = self.res3(x)
        #x = self.res4(x)

        x = F.avg_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    '''net = ClassficationModel()
    net = net.to('cuda')'''


if __name__=="__main__":
    main()
