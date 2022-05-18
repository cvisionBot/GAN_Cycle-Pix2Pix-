import torch
import functools
from torch import nn
from ..gen.gblocks import UNetSikpConnectionBlock, ResNetBlock
from ..initialize import weight_initialize


class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_downs=8, norm_layer=nn.BatchNorm2d,use_dropout=False):
        super(UNetGenerator, self).__init__()
        self.ngf = 64
        unet_block = UNetSikpConnectionBlock(self.ngf * 8, self.ngf * 8, in_channels=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UNetSikpConnectionBlock(self.ngf * 8, self.ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UNetSikpConnectionBlock(self.ngf * 4, self.ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSikpConnectionBlock(self.ngf * 2, self.ngf * 4, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSikpConnectionBlock(self.ngf, self.ngf * 2, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSikpConnectionBlock(out_channels, self.ngf, in_channels=in_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        output = self.model(input)
        return {'gen_pred' : output}


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=9, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(ResNetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.ngf = 64

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, self.ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(self.ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(self.ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResNetBlock(self.ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(self.ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(self.ngf, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return {'gen_pred' : output}



def CycleGAN_Generator(in_channels, out_channels, module_name):
    if module_name == 'unet': # unet 256
        model = UNetGenerator(in_channels=in_channels, out_channels=out_channels)
    elif module_name == 'resnet': # block = 9
        model = ResNetGenerator(in_channels=in_channels, out_channels=out_channels)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = CycleGAN_Generator(in_channels=3, out_channels=3, module_name='resnet')
    model(torch.rand(1, 3, 256, 256))