import torch
import pytorch_lightning as pl

from torch import nn
from models.loss.gan_loss import GAN_Loss
from utils.module_select import get_pix_optimizer


class PixGAN(pl.LightningModule):
    def __init__(self, generator, discriminator, cfg, epoch_length=None):
        super(PixGAN, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['generator', 'discriminator'])
        self.generator = generator
        self.discriminator = discriminator

        self.gan_loss = GAN_Loss(self.cfg)
        self.L1_loss = torch.nn.L1Loss()
        self.L1_loss_weight = self.cfg['lambda_L1']

    def forward(self, input):
        fake_B = self.generator(input)['gen_pred'] # input -> real_A Data
        return fake_B

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss_D, loss_G = self.opt_training_step(batch)
        self.log('Discriminator_loss', loss_D, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('GAN_loss', loss_G, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if optimizer_idx == 0:
            return loss_D
        if optimizer_idx == 1:
            return loss_G

    def opt_training_step(self, batch):
        # backward Discriminator Fake
        fake_B = self.generator(batch['real_A'])['gen_pred']
        fake_AB = torch.cat((batch['real_A'], fake_B), 1)
        pred_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)
        # backward Discriminator Real
        real_AB = torch.cat((batch['real_A'], batch['real_B']), 1)
        pred_real = self.discriminator(real_AB)
        loss_D_real = self.gan_loss(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        # backward Generator G(A)
        loss_G_GAN = self.gan_loss(pred_fake, True)
        # G(A) = B
        loss_G_L1 = self.L1_loss(fake_B, batch['real_B']) + self.L1_loss_weight
        loss_G = loss_G_GAN + loss_G_L1
        return loss_D, loss_G
    
    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim_d, optim_g = get_pix_optimizer(cfg['optimizer'],
            params=list(self.discriminator.parameters()) + list(self.generator.parameters()),
            **cfg['optimizer_options']
        )
        return [optim_d, optim_g]