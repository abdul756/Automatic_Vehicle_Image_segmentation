import torch
import torch.nn as nn
import torch.functional as F

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'device available', {device})
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,16,3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64,32,2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,20,3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


