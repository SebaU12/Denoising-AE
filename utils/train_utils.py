import torch 
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from utils.visual_utils import save_images_compare
import random

def validate_autoencoder(model, val_loader, criterion, device, epoch):
    total_loss_val = 0.0
    psnr_val_list = []
    ssim_val_list = []
    model.eval()
    
    flag = True
    cont = 0
    random_show = random.randint(0, len(val_loader))

    with torch.no_grad():
        for img, noisy_img, _ in val_loader:
            img, noisy_img = img.to(device), noisy_img.to(device)
            output = model(noisy_img)
            if flag == True and random_show == cont:
                save_images_compare(img, noisy_img, output, f'./recon/val/{epoch+1}')
                flag = False
            cont += 1
            loss = criterion(output, img)
            total_loss_val += loss.item()
            
            # Calculate PSNR and SSIM
            img_np  = img.cpu().numpy()
            output_np = output.cpu().numpy()
            psnr_batch = psnr(img_np[0].transpose(1, 2, 0), output_np[0].transpose(1, 2, 0), data_range=1)
            ssim_batch = ssim(img_np[0].transpose(1, 2, 0), output_np[0].transpose(1, 2, 0), data_range=1, multichannel=True, win_size=3)
            
            # Saving Metrics
            psnr_val_list.append(psnr_batch)
            ssim_val_list.append(ssim_batch)
            
    avg_loss_val = total_loss_val / len(val_loader)
    avg_psnr_val = sum(psnr_val_list) / len(psnr_val_list)
    avg_ssim_val = sum(ssim_val_list) / len(ssim_val_list)
    return avg_loss_val, avg_psnr_val, avg_ssim_val

def train_autoencoder(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    psnr_train_list_ae = []
    ssim_train_list_ae = []
    psnr_val_list_ae = []
    ssim_val_list_ae = []
    train_losses_ae = []
    val_losses_ae = []
    for epoch in range(num_epochs):
        i = 0
        model.train()
        epoch_train_loss = 0.0 
        psnr_train_batch_list = []
        ssim_train_batch_list = []
        flag = True
        for img, noisy_img, _ in train_loader:
            img, noisy_img = img.to(device), noisy_img.to(device)
            # Forward pass
            output = model(noisy_img)
            loss = criterion(output, img)

            if flag == True:
                save_images_compare(img, noisy_img, output, f'./recon/train/{epoch+1}')
                flag = False

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate PSNR and SSIM
            img_np = img.cpu().numpy()
            output_np = output.detach().cpu().numpy()
            psnr_batch = psnr(img_np[0].transpose(1, 2, 0), output_np[0].transpose(1, 2, 0), data_range=1)
            ssim_batch = ssim(img_np[0].transpose(1, 2, 0), output_np[0].transpose(1, 2, 0), data_range=1, multichannel=True, win_size=3)

            # Saving Metrics
            psnr_train_batch_list.append(psnr_batch)
            ssim_train_batch_list.append(ssim_batch)
            i += 1
            if (i + 1) % 300 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
            
        avg_loss = epoch_train_loss / len(train_loader)
        avg_psnr_train = sum(psnr_train_batch_list) / len(psnr_train_batch_list)
        avg_ssim_train = sum(ssim_train_batch_list) / len(ssim_train_batch_list)

        train_losses_ae.append(avg_loss)
        psnr_train_list_ae.append(avg_psnr_train)
        ssim_train_list_ae.append(avg_ssim_train)

        
        # Validation
        avg_val_loss, avg_psnr_val, avg_ssim_val = validate_autoencoder(model, val_loader, criterion, device, epoch)    
        val_losses_ae.append(avg_val_loss)
        psnr_val_list_ae.append(avg_psnr_val)
        ssim_val_list_ae.append(avg_ssim_val)

        print(f'Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return (train_losses_ae, val_losses_ae, psnr_train_list_ae, ssim_train_list_ae, psnr_val_list_ae, ssim_val_list_ae)