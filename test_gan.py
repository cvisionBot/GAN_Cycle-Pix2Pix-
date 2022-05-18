import os
import cv2
import argparse
import numpy as np

import torch
# from torchsummary import summary
from models.gen.generator import CycleGAN_Generator
from models.dis.discriminator import CycleGAN_Discriminator
from module.pix_gan import PixGAN
from module.cycle_gan import CycleGAN

from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs


def main(cfg, image_name, save):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (256, 256))
    image_inp = preprocess_input(image)
    image_inp = image_inp.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp=image_inp.cuda()

    # load trained model
    generator = CycleGAN_Generator(in_channels=cfg['gen_in_channels'], out_channels=cfg['gen_in_channels'], module_name=cfg['gen_module'])
    discriminator = CycleGAN_Discriminator(in_channels=cfg['dis_in_channels'], module_name=cfg['dis_module'])
    if torch.cuda.is_available():
        generator = generator.to('cuda')
        discriminator = discriminator.to('cuda')

    model_module = PixGAN.load_from_checkpoint(
        '/home/insig/GAN_Cycle_Pix2Pix/saved/Pix2Pix_GAN_Cityscapes/version_3/checkpoints/last.ckpt', generator=generator, discriminator=discriminator
    )

    model_module.eval()
    preds = model_module(image_inp)
    preds = preds * 255
    
    preds = preds.permute(0, 2, 3, 1)
    preds = torch.squeeze(preds)
    pred = preds.detach().cpu().numpy()

    cv2.imwrite("./inference/result/inference.png", pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--save', action='store_true',
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    main(cfg, './inference/sample/test01.jpg', args.save)