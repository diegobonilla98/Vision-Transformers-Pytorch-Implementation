import os
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import matplotlib
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
matplotlib.use('TKAgg')


class DogCats(Dataset):
    def __init__(self, size=(224, 224), dataset_path='/media/bonilla/My Book/DogsCats/data/train/*.jpg', augment=True):
        self.size = size
        self.transform = transforms.Compose([self.ToTensor(augment)])
        self.images_list = glob.glob(dataset_path)

    @staticmethod
    class ToTensor(object):
        def __init__(self, augment):
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.augment_pipeline = None
            if augment:
                self.augment_pipeline = iaa.Sequential([
                    iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                    iaa.Fliplr(0.5),
                    iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                    iaa.Sometimes(0.25, iaa.OneOf(
                        [
                            iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(0.1, size_percent=0.5)
                         ])),
                    iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                ])

        def __call__(self, sample):
            image, label = sample['image'], sample['class']
            if self.augment_pipeline is not None:
                image = self.augment_pipeline.augment_image(image)
            image = image.astype('float32') / 255.
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            image = self.normalize(image)
            return {'image': image,
                    'class': torch.from_numpy(label)}

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_list[idx])
        image = cv2.resize(image, self.size)[:, :, ::-1]

        y = np.array(int(self.images_list[idx].split(os.sep)[-1].startswith('cat')))

        sample = {'image': image, 'class': y}
        sample = self.transform(sample)
        sample['original'] = image.copy()
        return sample


if __name__ == '__main__':
    dl = DogCats()
    a = dl[0]
    print()
