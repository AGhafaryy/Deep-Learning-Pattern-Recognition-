import torch.nn as nn
from torchvision import models
import torch

#ToDO Fill in the __ values
class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.UnetMaxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        
        self.Up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.Up_conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.Up_conv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.Up_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.Up_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(64,self.n_class,kernel_size=1,stride=1,padding=0)



    def forward(self,x):
        x1 = self.Conv1(x)
        x2 = self.UnetMaxpool(x1)
        x2 = self.Conv2(x2)      
        x3 = self.UnetMaxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.UnetMaxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.UnetMaxpool(x4)
        x5 = self.Conv5(x5)

        
        x6 = self.Up5(x5)
        x6 = torch.cat((x4,x6),dim=1)    
        x6 = self.Up_conv5(x6)  
        x7 = self.Up4(x6)
        x7 = torch.cat((x3,x7),dim=1)
        x7 = self.Up_conv4(x7)
        x8 = self.Up3(x7)
        x8 = torch.cat((x2,x8),dim=1)
        x8 = self.Up_conv3(x8)
        x9 = self.Up2(x8)
        x9 = torch.cat((x1,x9),dim=1)
        x9 = self.Up_conv2(x9)
        
        score = self.classifier(x9)
        return score
        