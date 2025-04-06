import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define the Generator (hides the secret image in the cover image)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover, secret):
        x = torch.cat((cover, secret), dim=1).clone()  # Clone to prevent in-place modifications
        return self.model(x)

# Define the Discriminator (ensures encoded images look real)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Decoder (extracts the secret image from the encoded image)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator()
discriminator = Discriminator()
decoder = Decoder()

# Define loss functions
adv_loss = nn.BCELoss()
rec_loss = nn.MSELoss()

# Define optimizers
g_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
dec_opt = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training Loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
generator.to(device)
discriminator.to(device)
decoder.to(device)

for epoch in range(num_epochs):
    for i, (cover_images, _) in enumerate(dataloader):
        cover_images = cover_images.to(device)
        secret_images = cover_images[torch.randperm(cover_images.size(0))].clone().detach()  # Shuffle for secret images
        secret_images = secret_images.to(device)
        
        # Train Generator
        encoded_images = generator(cover_images, secret_images)
        disc_pred_fake = discriminator(encoded_images.clone().detach())
        g_loss = adv_loss(disc_pred_fake, torch.ones_like(disc_pred_fake))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()
        
        # Train Discriminator
        real_loss = adv_loss(discriminator(cover_images.clone().detach()), torch.ones_like(discriminator(cover_images.clone().detach())))
        fake_loss = adv_loss(discriminator(encoded_images.clone().detach()), torch.zeros_like(discriminator(encoded_images.clone().detach())))
        d_loss = (real_loss + fake_loss) / 2
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()
        
        # Train Decoder
        decoded_images = decoder(encoded_images.clone().detach())
        dec_loss = rec_loss(decoded_images, secret_images.clone().detach())
        dec_opt.zero_grad()
        dec_loss.backward()
        dec_opt.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}, Dec Loss: {dec_loss.item():.4f}")

print("Training Complete!")