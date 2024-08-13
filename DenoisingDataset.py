import torch 
from  torchvision import datasets
from utils.noise_utils import apply_random_noise

class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.dataset = datasets.SVHN(root=root, split=split, transform=transform, download=True)
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img_noisy = apply_random_noise(img, 1)
        return img, img_noisy, label

    def __len__(self):
        return len(self.dataset)
    
def create_denoising_datasets(transform):
    denoising_trainset = DenoisingDataset(root='./data', split='train', transform=transform)
    denoising_testset = DenoisingDataset(root='./data', split='test', transform=transform)
    return denoising_trainset, denoising_testset