import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
from DenoisingDataset import create_denoising_datasets
from utils.visual_utils import visualize_reconstruction
from model import Autoencoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 1

_ , test_dataset = create_denoising_datasets(transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
for img, noisy_img, _ in test_loader:
    img, noisy_img = img.to(device), noisy_img.to(device) 
    break

ae_conf = [3, 12, 3, 3, 64]
ae1 = Autoencoder(ae_conf[0], ae_conf[1], ae_conf[2], ae_conf[3], ae_conf[4], False).to(device)
ae1.load_state_dict(torch.load('model.pth'))
ae1.eval()
with torch.no_grad():
    output = ae1(noisy_img)
visualize_reconstruction(img, noisy_img, output, 1)
