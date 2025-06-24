import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Hyperparameters
lr = 0.0002
batch_size = 64
epochs = 100
noise_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist = datasets.MNIST(root='.', download=True, transform=transform)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Models
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

# Optimizers and Loss
loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr)
opt_D = optim.Adam(D.parameters(), lr=lr)

# Storage
losses_G, losses_D = [], []
fixed_noise = torch.randn(16, noise_dim, device=device)

# Training
for epoch in range(epochs + 1):
    loss_G_epoch, loss_D_epoch = 0, 0
    for real, _ in loader:
        real = real.to(device)
        batch_size = real.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = G(z)
        D_real = D(real)
        D_fake = D(fake.detach())

        loss_D = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = G(z)
        D_fake = D(fake)
        loss_G = loss_fn(D_fake, real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loss_G_epoch += loss_G.item()
        loss_D_epoch += loss_D.item()

    # Save losses
    losses_G.append(loss_G_epoch / len(loader))
    losses_D.append(loss_D_epoch / len(loader))

    # Show samples
    if epoch in [0, 50, 100]:
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
        grid = np.transpose(torchvision.utils.make_grid(samples, nrow=4, normalize=True), (1, 2, 0))
        plt.imshow(grid)
        plt.title(f"Epoch {epoch}")
        plt.axis("off")
        plt.savefig(f"generated_epoch_{epoch}.png")
        plt.close()

    print(f"Epoch {epoch} | Loss D: {loss_D_epoch:.4f}, Loss G: {loss_G_epoch:.4f}")

# Plot losses
plt.plot(losses_G, label="Generator Loss")
plt.plot(losses_D, label="Discriminator Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("GAN Training Losses")
plt.savefig("gan_losses.png")
plt.show()
