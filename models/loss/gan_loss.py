import math
import torch
import numpy as np
from torch import nn


class GAN_Loss(nn.Module):
    def __init__(self, cfg, target_real_label=1.0, target_fake_label=0.0):
        super(GAN_Loss, self).__init__()
        self.cfg = cfg
        self.gan_mode = self.cfg['gan_mode']
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('not implemented gan mode')

    def forward(self, prediction, target_is_real):
        '''
        prediction (tensor) - discriminator (output)
        target_is_real (bool) - ground truth label (real or fake image)
        '''
        prediction = prediction['dis_pred']
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
