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


def main(cfg, image_name_A, image_name_B, save):
    image_A = cv2.imread(image_name_A)
    image_A = cv2.resize(image_A, (256, 256))
    image_inp_A = preprocess_input(image_A)
    image_inp_A = image_inp_A.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp_A=image_inp_A.cuda()

    image_B = cv2.imread(image_name_B)
    image_B = cv2.resize(image_B, (256, 256))
    image_inp_B = preprocess_input(image_B)
    image_inp_B = image_inp_B.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp_B=image_inp_B.cuda()

    # load trained model
    generator_basestyle = CycleGAN_Generator(in_channels=cfg['gen_in_channels'], out_channels=cfg['gen_in_channels'], module_name=cfg['gen_module'])
    discriminator_basestyle = CycleGAN_Discriminator(in_channels=cfg['dis_in_channels'], module_name=cfg['dis_module'])
    generator_stylebase = CycleGAN_Generator(in_channels=cfg['gen_in_channels'], out_channels=cfg['gen_in_channels'], module_name=cfg['gen_module'])
    discriminator_stylebase = CycleGAN_Discriminator(in_channels=cfg['dis_in_channels'], module_name=cfg['dis_module'])
    if torch.cuda.is_available():
        generator_basestyle = generator_basestyle.to('cuda')
        generator_stylebase = generator_stylebase.to('cuda')
        discriminator_basestyle = discriminator_basestyle.to('cuda')
        discriminator_stylebase = discriminator_stylebase.to('cuda')

    model_module = CycleGAN.load_from_checkpoint(
        '/home/insig/GAN_Cycle_Pix2Pix/saved/Cycle_GAN_Apple2Orange/version_1/checkpoints/last.ckpt', 
            G_basestyle=generator_basestyle, G_stylebase=generator_stylebase, D_base=discriminator_basestyle , D_style=discriminator_stylebase
    )

    images = {'real_A': image_inp_A, 'real_B': image_inp_B}
    model_module.eval()
    preds = model_module(images)
    preds_A = preds['domain_A']
    preds_B = preds['domain_B']
    # print(preds_A)
    preds_A = preds_A * 255
    preds_B = preds_B * 255

    preds_A = preds_A.permute(0, 2, 3, 1)
    preds_A = torch.squeeze(preds_A)

    preds_B = preds_B.permute(0, 2, 3, 1)
    preds_B = torch.squeeze(preds_B)

    preds_A = preds_A.detach().cpu().numpy()
    preds_B = preds_B.detach().cpu().numpy()

    cv2.imwrite("./inference/result/A_inference.png", preds_A)
    cv2.imwrite("./inference/result/B_inference.png", preds_B)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--save', action='store_true',
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    main(cfg, image_name_A='/home/insig/GAN_Cycle_Pix2Pix/dataset/datasets/apple2orange/testA/n07740461_240.jpg', 
                image_name_B='/home/insig/GAN_Cycle_Pix2Pix/dataset/datasets/apple2orange/testB/n07749192_61.jpg', save=args.save)

                #testA/n07740461_20.jpg