import torch
import torch.nn as nn

from .Convolution import Convolution

class Deconvolution(nn.Module):
    def __init__(self,input_channel,skip_input_channel):
        super().__init__()
        self.in_channels = input_channel
        self.skip_in_channels = skip_input_channel
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=input_channel,out_channels=skip_input_channel,kernel_size=4,stride=4),
            nn.BatchNorm1d(skip_input_channel)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(skip_input_channel, skip_input_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(skip_input_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self,x,skip):
        if x.shape[1]!=self.in_channels:
            x = x.transpose(2,1)
        if skip.shape[1]!=self.skip_in_channels:
            skip = skip.transpose(2,1)
        x_up = self.deconv(x)
        x_skip_joined = x_up + skip
        output = self.conv(x_skip_joined)
        if output.shape[1]==self.skip_in_channels:
            output = output.transpose(2,1)

        return output
    
# if __name__ == "__main__":
#     deconv = Deconvolution(input_channel=768,skip_input_channel=384)
#     input = torch.rand(2,49,768)

#     skip = torch.rand(2,196,384)

#     output = deconv(input,skip)


#     print(output.shape) # 2, 196, 384