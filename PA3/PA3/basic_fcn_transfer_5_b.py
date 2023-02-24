import torch.nn as nn
from torchvision import models

#ToDO Fill in the __ values
class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.relu = nn.ReLU(inplace=True) 
        self.model = models.resnet18(pretrained=True)  
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
        
#TODO Complete the forward pass
    def forward(self, x):
        x5 = self.model(x)
        x6 = self.deconv1(x5)
        x6 = self.bn1(self.relu(x6))
        x7 = self.deconv2(x6)
        x7 = self.bn2(self.relu(x7))
        x8 = self.deconv3(x7)
        x8 = self.bn3(self.relu(x8))
        x9 = self.deconv4(x8)
        x9 = self.bn4(self.relu(x9))
        x10 = self.deconv5(x9)
        x10 = self.bn5(self.relu(x10))

        score = self.classifier(x10)
        
        return score