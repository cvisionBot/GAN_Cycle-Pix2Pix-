import cv2
import glob
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from dataset.GAN.pix_utils import pix_collater, visaulize

class City_pixelDataset(Dataset):
    def __init__(self, transforms, path=None):
        super(City_pixelDataset, self).__init__()
        self.transforms = transforms
        self.image_A = glob.glob(path + '/trainA/*.jpg')
        self.image_B = glob.glob(path + '/trainB/*.jpg')

    def __len__(self):
        return len(self.image_A)

    def __getitem__(self, index):
        A_image_file = self.image_A[index]
        A_image = cv2.imread(A_image_file)
        A_transformed = self.transforms(image=A_image)

        B_image_file = self.image_B[index]#self.load_real_data_file(A_image_file)
        B_image = cv2.imread(B_image_file)
        B_transformed = self.transforms(image=B_image)
        return A_transformed, B_transformed

    def load_real_data_file(self, img_file):
        img_file = img_file.replace('A', 'B')
        return img_file


class GAN_Format(pl.LightningDataModule):
    def __init__(self, train_path, workers, train_transforms, batch_size=None):
        super().__init__()
        self.train_path = train_path
        self.train_transforms = train_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(City_pixelDataset(
            transforms=self.train_transforms,
            path=self.train_path),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=pix_collater)
    

if __name__ == '__main__':
    """
    Data Loader 테스트 코드
    python -m dataset.GAN.pix2pix_format
    """
    import albumentations
    import albumentations.pytorch

    train_transforms = albumentations.Compose([
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])

    loader = DataLoader(City_pixelDataset(
        transforms=train_transforms, path='/home/insig/GAN_Cycle_Pix2Pix/dataset/datasets/apple2orange'),
        batch_size=8, shuffle=True, collate_fn=pix_collater)
    
    for batch, sample in enumerate(loader):
        imgs_A = sample['real_A']
        imgs_B = sample['real_B']
        visaulize(imgs_A, imgs_B)
        break

    