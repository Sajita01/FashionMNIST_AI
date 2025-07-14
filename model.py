import torch.nn as nn

class GANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(GANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0),  # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)  # (B, latent_dim, 1, 1)
        return self.model(x)


import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),  # (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),  # (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)  # <- match key: 'main.X.Y'


