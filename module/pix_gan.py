import torch
import pytorch_lightning as pl

from torch import nn
from torch import optim
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
        self.mae = torch.nn.L1Loss()
        self.reconstr_weight = self.cfg['lambda_L1']

    def forward(self, input):
        fake_B = self.generator(input)['gen_pred'] # input -> real_A Data
        return fake_B

    def training_step(self, batch, batch_idx, optimizer_idx):
        A_img = batch['real_A']
        B_img = batch['real_B']

        if optimizer_idx == 0:
            fake_B = self.geneartor(A_img)['gen_pred']
            fake_AB = torch.cat((A_img, fake_B), 1)
            pred_fake = self.dicriminator(fake_AB)['dis_pred']
            loss_G_val = self.gan_loss(pred_fake, True)
            Loss_G_idt = self.mae(fake_B, B_img) + self.reconstr_weight
            loss_G = loss_G_val + Loss_G_idt
            self.log('Generator_loss', loss_G, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss_G
        elif optimizer_idx == 1:
            fake_B = self.generator(A_img)['pred']
            fake_AB = torch.cat(A_img, fake_B, 1)
            pred_fake = self.discriminator(fake_AB)['dis_pred']
            loss_D_fake = self.gan_loss(pred_fake, False)

            real_AB = torch.cat((A_img, B_img), 1)
            pred_real = self.discriminator(real_AB)['dis_pred']
            loss_D_real = self.gan_loss(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            self.log('Discriminator_loss', loss_G, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss_D
    
    def configure_optimizers(self):
        cfg = self.hparams.cfg
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
        return [self.generator_optimizer, self.discriminator_optimizer], []