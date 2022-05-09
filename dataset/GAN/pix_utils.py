import cv2
import torch
import numpy as np


def pix_collater(data):
    A = []
    B = []
    batch_size = len(data)

    for i in range(batch_size):
        A.append(data[i][0])
    for j in range(batch_size):
        B.append(data[j][1])

    imgs_A = [s['image'] for s in A]
    imgs_B = [s['image'] for s in B]
    return {'real_A' : torch.stack(imgs_A), 'real_B' : torch.stack(imgs_B)}


def visaulize(image_A, image_B, batch_idx=0):
    print(image_A.shape)
    print(image_B.shape)
    img_A = image_A[batch_idx].numpy()
    img_B = image_B[batch_idx].numpy()
    img_A = (np.transpose(img_A, (1, 2, 0)) * 255.).astype(np.uint8).copy()
    img_B = (np.transpose(img_B, (1, 2, 0)) * 255.).astype(np.uint8).copy()

    cv2.imwrite('./dataset/GAN/real_A.png', img_A)
    cv2.imwrite('./dataset/GAN/real_B.png', img_B)