from random import choice, uniform
import torch

def add_gaussian_noise(image, mean=0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.clone()
    num_salt = int(salt_prob * image.numel())
    num_pepper = int(pepper_prob * image.numel())
    
    salt_coords = [torch.randint(0, i - 1, (num_salt,)) for i in image.shape]
    pepper_coords = [torch.randint(0, i - 1, (num_pepper,)) for i in image.shape]
    
    noisy_image[salt_coords] = 1
    noisy_image[pepper_coords] = 0
    
    return noisy_image

def add_poisson_noise(image, scale=1):
    noise = torch.poisson(image * scale) / scale
    return torch.clamp(noise, 0, 1)

def add_speckle_noise(image, noise_factor=0.1):
    noise = torch.randn_like(image) * noise_factor
    noisy_image = image + image * noise
    return torch.clamp(noisy_image, 0, 1)

def apply_random_noise(image, noise_level=1.0, debug=0):
    noise_type = choice(['gaussian', 'salt_and_pepper', 'speckle', 'poisson'])
    if(debug== 1): print(noise_type)
    if noise_type == 'gaussian':
        std = uniform(0.01, 0.25) * noise_level
        return add_gaussian_noise(image, std=std)
    elif noise_type == 'salt_and_pepper':
        salt_prob = uniform(0.01, 0.15) * noise_level
        pepper_prob = uniform(0.01, 0.15) * noise_level
        return add_salt_and_pepper_noise(image, salt_prob=salt_prob, pepper_prob=pepper_prob)
    elif noise_type == 'poisson':
        scale = uniform(18, 25) * noise_level
        return add_poisson_noise(image, scale=scale)
    elif noise_type == 'speckle':
        noise_factor = uniform(0.01, 0.3) * noise_level
        return add_speckle_noise(image, noise_factor=noise_factor)