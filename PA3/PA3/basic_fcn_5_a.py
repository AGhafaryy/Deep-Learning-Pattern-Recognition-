import torch.nn as nn
import sys

debug = False

def printf(*tp):
    if debug:
        print(" ".join(map(str,tp)))

#ToDO Fill in the __ values
class FCN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.LeakyReLU(0.01, inplace=False)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=1, dilation=1) # 108
        self.bnd1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1) # 54
        self.bnd2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1) # 27
        self.bnd3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1) # 14
        self.bnd4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1) # 7
        self.bnd5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1) # 4
        self.bnd6 = nn.BatchNorm2d(1024)
        
#         self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, dilation=1) # 2
#         self.bnd7 = nn.BatchNorm2d(2048)
        
#         self.conv8 = nn.Conv2d(2048, 4096, kernel_size=3, stride=2, padding=1, dilation=1) # 1
#         self.bnd8 = nn.BatchNorm2d(4096)
        
        
        
#         self.deconv1 = nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # 1
#         self.bn1 = nn.BatchNorm2d(2048)
        
        
#         self.deconv2 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn2 = nn.BatchNorm2d(1024)
        
        self.deconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.deconv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.deconv6 = nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=1, dilation=1, output_padding=0)
        self.bn6 = nn.BatchNorm2d(64)
        
        self.deconv7 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        
        self.deconv8 = nn.ConvTranspose2d(32, 32, kernel_size=12, stride=2, padding=1, dilation=1, output_padding=0)
        self.bn8 = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
        
        
        

    def forward(self, x):
        # 224
        
        x = self.bnd1(self.relu(self.conv1(x))) # 108
        printf("conv1:",x.shape)        
        x = self.bnd2(self.relu(self.conv2(x))) # 54
        printf("conv2:",x.shape)
        x, ind1 = self.pool(x) # 27
        printf("pool:",x.shape)
        
        x = self.bnd3(self.relu(self.conv3(x))) # 14
        printf("conv3:",x.shape)
        x = self.bnd4(self.relu(self.conv4(x))) # 7
        printf("conv4:",x.shape)
        x, ind2 = self.pool(x) # 4
        printf("pool:",x.shape)
        
        x = self.bnd5(self.relu(self.conv5(x))) # 2
        printf("conv5:",x.shape)
        x = self.bnd6(self.relu(self.conv6(x))) # 1
        printf("conv6:",x.shape)
        
        # x, ind1 = self.pool(x)
        # printf("pool:",x.shape)
        # x = self.bnd7(self.relu(self.conv7(x))) # 2
        # printf("conv7:",x.shape)
        # x = self.bnd8(self.relu(self.conv8(x))) # 1
        # printf("conv8:",x.shape)
        # Complete the forward function for the rest of the encoder
        # sys.exit("bhadwi")

#         y = self.bn1(self.relu(self.deconv1(x))) # 2
#         printf("deconv1:",y.shape)
        
#         y = self.bn2(self.relu(self.deconv2(y))) # 4
#         printf("deconv2:",y.shape)

        y = self.bn3(self.relu(self.deconv3(x))) # 2
        printf("deconv3:",y.shape)
        
        y = self.bn4(self.relu(self.deconv4(y))) # 3
        printf("deconv4:",y.shape)
        
        y = self.unpool(y, ind2) # 7
        printf("unpool:",y.shape)
        
        
        y = self.bn5(self.relu(self.deconv5(y))) # 14
        printf("deconv5:",y.shape)
        
        y = self.bn6(self.relu(self.deconv6(y))) # 27
        printf("deconv6:",y.shape)
        
        printf(ind1.shape, y.shape)
        y = self.unpool(y, ind1) # 54
        printf("unpool:",y.shape)
        
        y = self.bn7(self.relu(self.deconv7(y))) # 108
        printf("deconv7:",y.shape)
        
        y = self.bn8(self.relu(self.deconv8(y))) # 224
        printf("deconv8:",y.shape)
        # Complete the forward function for the rest of the decoder
        
        # sys.exit("bhadwi")

        score = self.classifier(y)
        printf("classifier:",score.shape)
        
        if debug:
            sys.exit("lawdi")
        
        return score  # size=(N, n_class, H, W)
        
        # score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)