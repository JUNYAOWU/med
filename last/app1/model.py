import torch
import torch.nn as nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, inp, oup, stride=1,use_cv1=False):
        super(Residual, self).__init__()
        self.Mish=nn.Mish()
        self.conv1 = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=3,  padding=1)
        self.bn1=nn.BatchNorm2d(oup)
        self.bn2=nn.BatchNorm2d(oup)
        if use_cv1:
            self.conv3 = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=stride, padding=0)
        else:
            self.conv3=None

    def forward(self, x):

        y=self.Mish(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3 :
            x=self.conv3(x)
        y=self.Mish(y+x)
        return y

class ResNet(nn.Module):
    def __init__(self,Residual):
        super(ResNet, self).__init__()
        #其他块
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.Mish(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2=nn.Sequential(
            Residual(64,64,use_cv1=False,stride=1),
            Residual(64,64,use_cv1=False,stride=1)
        )
        self.b3=nn.Sequential(
            Residual(64,128,use_cv1=True,stride=1),
            Residual(128,128,use_cv1=False,stride=1)
        )
        self.b4=nn.Sequential(
            Residual(128,256,use_cv1=True,stride=2),
            Residual(256,256,use_cv1=False,stride=1)
        )
        self.b5=nn.Sequential(
            Residual(256,512,use_cv1=True,stride=2),
            Residual(512,512,use_cv1=True,stride=1)
        )
        self.b6=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=10))

    def forward(self, x):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        x=self.b6(x)
        return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Residual).to(device)
    print(summary(model,(3,224,224)))
