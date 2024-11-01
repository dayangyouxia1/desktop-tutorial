import PIL.Image as Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings


def imshow(img_path, transform):
    """
    Function to show data augmentation
    Param img_path: path of the image
    Param transform: data augmentation technique to apply
    """
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title(f'Original image {img.size}')
    ax[0].imshow(img)
    img = transform(img)
    ax[1].set_title(f'Transformed image {img.size}')
    ax[1].imshow(img)
    plt.show()


path = 'kitten.jpg'
transform = transforms.Resize((224, 224))
imshow(path, transform)

transform = transforms.CenterCrop((224, 224))
imshow(path, transform)

transform = transforms.RandomResizedCrop((100, 300))
imshow(path, transform)

transform = transforms.RandomHorizontalFlip()
imshow(path, transform)

transform = transforms.Pad((50, 50, 50, 50))
imshow(path, transform)

transform = transforms.RandomRotation(15)
imshow(path, transform)

transform = transforms.RandomAffine(1, translate=(0.5, 0.5), scale=(1, 1), shear=(1,1), fill=(256,256,256))
# transform = transforms.RandomAffine(1, translate=(0.5, 0.5), scale=(1, 1), shear=(1,1), fillcolor=(256,256,256))
imshow(path, transform)

transform = transforms.GaussianBlur(7, 3)
imshow(path, transform)

transform = transforms.Grayscale(num_output_channels=3)
imshow(path, transform)

transform = transforms.ColorJitter(brightness=2)
imshow(path, transform)

transform = transforms.ColorJitter(contrast=2)
imshow(path, transform)

transform = transforms.ColorJitter(saturation=20)
imshow(path, transform)

# transform = transforms.ColorJitter(hue=2)
transform = transforms.ColorJitter(hue=0.2)
imshow(path, transform)




