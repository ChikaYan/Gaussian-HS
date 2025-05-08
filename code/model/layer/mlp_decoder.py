import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder2D(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4], out_rescale=1, act_fn=None, init_zero_output=False):
        super(MLPDecoder2D, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.out_rescale = out_rescale
        self.act_fn = act_fn
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_linear = nn.Linear(W, output_ch)

        if init_zero_output:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)
            torch.nn.init.constant_(self.output_linear.weight, 0.0)

    def forward(self, x):
        
        input = x.permute([0,2,3,1]).reshape([-1, self.input_ch])

        h = input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input, h], -1)

        outputs = self.output_linear(h)
        outputs = outputs.reshape([x.shape[0], *x.shape[2:], -1]).permute([0,3,1,2]) * self.out_rescale
        if self.act_fn is not None:
            outputs = self.act_fn(outputs)
        return outputs
