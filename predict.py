import torch
from archs import NestedUNet
import cv2
import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

Path = 'models/meibomian_gland_NestedUNet_woDS/model.pth'
model = NestedUNet(1)
model.load_state_dict(torch.load(Path))
img = cv2.imread('1.JPG')
train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        albu.Resize(96, 96),
        transforms.Normalize(),
    ])
img_tensor = train_transform(image=img)
output = model(img)
cv2.imwrite('2.png', output)