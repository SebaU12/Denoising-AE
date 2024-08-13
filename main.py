import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from DenoisingDataset import create_denoising_datasets
from utils.train_utils import train_autoencoder
from utils.visual_utils import save_loss_metric, save_reconstruction_metric, show_images
from model import Autoencoder

torch.manual_seed(20)

batch_size = 32
num_epochs = 15
ae_conf = [3, 12, 3, 3, 64]

def show_details(data_loader):
    img, noisy_img, _ = next(iter(data_loader))
    image = img[0]  
    noisy_image= noisy_img[0] 
    print(f"Min value: {noisy_image.min().item()}")
    print(f"Max value: {noisy_image.max().item()}")
    print(f"Size values: {noisy_image.size()}")
    show_images(image, noisy_image)

def start_train(train_loader, test_loader, device):
    print(f'Start Training with {device}')
    model = Autoencoder(input_channels=ae_conf[0], num_layers=ae_conf[1], 
                        stride_interval=ae_conf[2], channels_interval=ae_conf[3], 
                        base_channel=ae_conf[4], debug=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.MSELoss()
    train_results = train_autoencoder(model, train_loader, test_loader, num_epochs, criterion, optimizer, device)
    torch.cuda.empty_cache()
    train_losses  = train_results[0]
    val_losses = train_results[1]
    psnr_train_list = train_results[2]
    ssim_train_list = train_results[3]
    psnr_val_list = train_results[4]
    ssim_val_list = train_results[5]
    print("Saving Metrics")
    save_loss_metric(train_losses, val_losses)
    save_reconstruction_metric(psnr_train_list, ssim_train_list, psnr_val_list, ssim_val_list)
    print("Saving model")
    torch.save(model.state_dict(), 'model.pth')

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset, test_dataset = create_denoising_datasets(transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    show_details(train_loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    summary(Autoencoder(ae_conf[0], ae_conf[1], ae_conf[2], ae_conf[3], ae_conf[4], True), torch.zeros((batch_size, 3, 32, 32)), show_input=True, print_summary=True)
    start_train(train_loader, test_loader, device)

if __name__ == '__main__':
    main()