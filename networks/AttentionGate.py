import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self,gate_input_channel,skip_input_channel):
        super().__init__()
        self.gate_input_channel = gate_input_channel
        self.skip_input_channel = skip_input_channel
        self.Wg = nn.Sequential(
            nn.ConvTranspose1d(in_channels=gate_input_channel,out_channels=skip_input_channel,kernel_size=4,stride=4),
            nn.BatchNorm1d(skip_input_channel)
        )
        self.Ws = nn.Sequential(
            nn.Conv1d(in_channels=skip_input_channel,out_channels=skip_input_channel,kernel_size=3,padding=1),
            nn.BatchNorm1d(skip_input_channel)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv1d(skip_input_channel, skip_input_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):   # g is gating signal from lower dimension, s is skip connection
        if g.shape[1]!=self.gate_input_channel:
            g = g.transpose(2,1)
        if s.shape[1]!=self.skip_input_channel:
            s = s.transpose(2,1)
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        out = out * s
        out = out.transpose(2,1)
        return out

# if __name__ == "__main__":
#     att = AttentionGate(768,384)

#     g = torch.rand(2,49,768)
#     s = torch.rand(2,196,384)

#     skip = att(g,s)

#     print(skip.shape)  # 2, 196, 384