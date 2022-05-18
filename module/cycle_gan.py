import torch
import pytorch_lightning as pl

from torch import nn
from torch import optim
from models.loss.gan_loss import GAN_Loss
from utils.module_select import get_cycle_optimizer


class CycleGAN(pl.LightningModule):
    def __init__(self, G_basestyle, G_stylebase, D_base, D_style, cfg, epoch_length=None):
        super(CycleGAN, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['G_basestyle', 'G_stylebase', 'D_base', 'D_style'])
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style=D_style

        self.gan_loss = GAN_Loss(self.cfg)
        self.mae = nn.L1Loss()
        self.reconstr_weight = self.cfg['lambda_L1']
        self.id_w = self.cfg['lambda_idt']

        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []
        
    def forward(self, input):
        fake_B = self.G_basestyle(input['real_B'])['gen_pred']
        rec_A = self.G_stylebase(fake_B)['gen_pred']
        fake_A = self.G_stylebase(input['real_A'])['gen_pred']
        rec_B = self.G_basestyle(fake_A)['gen_pred']
        return {'domain_A': rec_A, 'domain_B': rec_B}
        # return {'domain_A' : fake_A, 'domain_B' : fake_B}

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img = batch['real_A']
        style_img = batch['real_B']
        
        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            val_base = self.gan_loss(self.D_base(self.G_stylebase(style_img)['gen_pred'])['dis_pred'], True)
            val_style = self.gan_loss(self.D_style(self.G_basestyle(base_img)['gen_pred'])['dis_pred'], True)
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(self.G_stylebase(self.G_basestyle(base_img)['gen_pred'])['gen_pred'], base_img)
            reconstr_style = self.mae(self.G_basestyle(self.G_stylebase(style_img)['gen_pred'])['gen_pred'], style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            id_base = self.mae(self.G_stylebase(base_img)['gen_pred'], base_img)
            id_style = self.mae(self.G_basestyle(style_img)['gen_pred'], style_img)
            id_loss = (id_base + id_style) / 2

            # Loss weight
            G_loss = val_loss + self.reconstr_weight * reconstr_loss + self.id_w * id_loss
            self.log('Generator_loss', G_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return {'loss':G_loss, 'validity':val_loss, 'reconstr':reconstr_loss, 'identity':id_loss}

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            D_base_gen_loss = self.gan_loss(self.D_base(self.G_stylebase(style_img)['gen_pred'])['dis_pred'], False)
            D_style_gen_loss = self.gan_loss(self.D_style(self.G_basestyle(base_img)['gen_pred'])['dis_pred'], False)
            D_base_valid_loss = self.gan_loss(self.D_base(base_img)['dis_pred'], True)
            D_style_valid_loss = self.gan_loss(self.D_style(style_img)['dis_pred'], True)

            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2

            # Loss weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3
            self.log('Discriminator_loss', D_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return {'loss':D_loss}

    def training_epoch_end(self, outputs):
        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 4 for i in range(4)])
        G_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        D_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in [2, 3]])
        validity = sum([torch.stack([x['validity'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        reconstr = sum([torch.stack([x['reconstr'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        identity = sum([torch.stack([x['identity'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
            
        self.losses.append(avg_loss)
        self.G_mean_losses.append(G_mean_loss)
        self.D_mean_losses.append(D_mean_loss)
        self.validity.append(validity)
        self.reconstr.append(reconstr)
        self.identity.append(identity)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
        self.d_base_optimizer = optim.Adam(self.D_base.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
        self.d_style_optimizer = optim.Adam(self.D_style.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer, self.d_style_optimizer], []

