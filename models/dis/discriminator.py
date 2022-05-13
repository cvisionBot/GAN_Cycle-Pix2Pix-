import torch
import functools
from torch import nn
from ..initialize import weight_initialize


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        self.ndf = 128
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, self.ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(self.ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(self.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(self.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return {'dis_pred' : output}


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        self.ndf = 64
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(in_channels, self.ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        output = self.net(input)
        return {'dis_pred' : output}



def CycleGAN_Discriminator(in_channels, module_name):
    if module_name == 'path_gan':
        model = NLayerDiscriminator(in_channels=in_channels)
    else:
        model = PixelDiscriminator(in_channels=in_channels)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = CycleGAN_Discriminator(in_channels=6, module_name='none')
    model(torch.rand(1, 6, 256, 256))

