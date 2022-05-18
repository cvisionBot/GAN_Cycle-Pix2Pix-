import argparse
from dis import dis
from logging import raiseExceptions
import platform
from sys import implementation
import albumentations
import albumentations.pytorch
import pytorch_lightning as pl

from utils.utility import make_model_name
from utils.yaml_helper import get_train_configs

from dataset.GAN.gan_format import GAN_Format
from models.gen.generator import CycleGAN_Generator
from models.dis.discriminator import CycleGAN_Discriminator
from module.pix_gan import PixGAN
from module.cycle_gan import CycleGAN

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def train(cfg, ckpt=None):
    input_size=cfg['input_size']
    train_transforms = albumentations.Compose([
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])
    data_module = None
    model_module = None

    data_module = GAN_Format(
            train_path=cfg['data_path'], workers=cfg['workers'],
            train_transforms=train_transforms, batch_size=cfg['batch_size']
    )
    generator = CycleGAN_Generator(in_channels=cfg['gen_in_channels'], out_channels=cfg['gen_in_channels'], module_name=cfg['gen_module'])
    discriminator = CycleGAN_Discriminator(in_channels=cfg['dis_in_channels'], module_name=cfg['dis_module'])
    if cfg['model'] == 'Pix2Pix_GAN':
        model_module = PixGAN(generator=generator, discriminator=discriminator, cfg=cfg, epoch_length=data_module.train_dataloader().__len__())
    elif cfg['model'] == 'Cycle_GAN':
        generator_basestyle = CycleGAN_Generator(in_channels=cfg['gen_in_channels'], out_channels=cfg['gen_in_channels'], module_name=cfg['gen_module'])
        discriminator_basestyle = CycleGAN_Discriminator(in_channels=cfg['dis_in_channels'], module_name=cfg['dis_module'])
        generator_stylebase = CycleGAN_Generator(in_channels=cfg['gen_in_channels'], out_channels=cfg['gen_in_channels'], module_name=cfg['gen_module'])
        discriminator_stylebase = CycleGAN_Discriminator(in_channels=cfg['dis_in_channels'], module_name=cfg['dis_module'])
        model_module = CycleGAN(G_basestyle=generator_basestyle, G_stylebase=generator_stylebase, D_base=discriminator_basestyle, D_style=discriminator_stylebase,
                                     cfg=cfg, epoch_length=data_module.train_dataloader().__len__())
    else:
        raise NotImplementedError('Out of GAN Method!')

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(monitor='Generator_loss', save_last=True, every_n_epochs=cfg['save_freq']),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], make_model_name(cfg)),
        gpus=cfg['gpus'],
        accelerator='ddp' if platform.system() != 'Windows' else None,
        plugins=DDPPlugin() if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        resume_from_checkpoint=ckpt,
        # **cfg['trainer_options'],
    )
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Train config file')
    parser.add_argument('--ckpt', required=False, type=str, help='Train checkpoint')
    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    train(cfg, args.ckpt)