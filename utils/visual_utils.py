import matplotlib.pyplot as plt
import numpy as np

def show_images(original_image, noisy_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(original_image.numpy(), (1, 2, 0)))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(noisy_image.numpy(), (1, 2, 0)))
    plt.title('Noisy Image')
    plt.axis('off')
    plt.show()

def save_loss_metric(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation Loss for AE')
    plt.legend()
    plt.savefig(f'loss_metric_ae.png')

def save_reconstruction_metric(psnr_train_list, ssim_train_list, psnr_val_list, ssim_val_list):
    epochs = range(1, len(psnr_train_list) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, psnr_train_list, label='Train PSNR')
    plt.plot(epochs, psnr_val_list, label='Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title(f'PSNR Evolution for AE')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ssim_train_list, label='Train SSIM')
    plt.plot(epochs, ssim_val_list, label='Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title(f'SSIM Evolution for AE')
    plt.savefig(f'metrics_evolution_ae.png')

def visualize_reconstruction(img, noisy_img, reconstructed_img, ae_num):
    img = img.cpu().detach()
    noisy_img = noisy_img.cpu().detach()
    reconstructed_img = reconstructed_img.cpu().detach()
    
    img = img.numpy().transpose(0, 2, 3, 1)
    noisy_img = noisy_img.numpy().transpose(0, 2, 3, 1)
    reconstructed_img = reconstructed_img.numpy().transpose(0, 2, 3, 1)
    
    # Show plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img[0])
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(noisy_img[0])
    ax[1].set_title(f'Noisy Image {ae_num}')
    ax[1].axis('off')
    
    ax[2].imshow(reconstructed_img[0])
    ax[2].set_title(f'Reconstructed Image {ae_num}')
    ax[2].axis('off')
    plt.show()

def save_images_compare(img, noisy_img, reconstructed_img, name):
    img = img.cpu().detach()
    noisy_img = noisy_img.cpu().detach()
    reconstructed_img = reconstructed_img.cpu().detach()
    
    img = img.numpy().transpose(0, 2, 3, 1)
    noisy_img = noisy_img.numpy().transpose(0, 2, 3, 1)
    reconstructed_img = reconstructed_img.numpy().transpose(0, 2, 3, 1)
    
    # Save plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img[0])
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(noisy_img[0])
    ax[1].set_title(f'Noisy Image')
    ax[1].axis('off')
    
    ax[2].imshow(reconstructed_img[0])
    ax[2].set_title(f'Reconstructed Image')
    ax[2].axis('off')
    plt.savefig(f'{name}.png')
    plt.close(fig)