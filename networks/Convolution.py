import torch
import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
         
    def forward(self,x,maxpool=True):
        if x.shape[1]!=self.in_channels:
            x = x.transpose(2,1)
        x_conv = self.conv(x)
        if maxpool==True:
            x_conv = self.maxpool(x_conv)

        if x_conv.shape[1]==self.out_channels:
            x_conv = x_conv.transpose(2,1)

        return x_conv
    
# if __name__=="__main__":
#     conv = Convolution(96,192)
#     x = torch.rand(2,3136,96)
#     x_conv = conv(x)

#     print(x_conv.shape) # 2,784,192