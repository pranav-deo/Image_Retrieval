import torch
import torch.nn as nn

Num_channels = 8
inner_channels = 128


class AE(nn.Module):
    """AE for mulitask hashing, reconstruction"""

    def __init__(self, K):
        super(AE, self).__init__()

        # Encoder:
        self.e_conv = nn.ModuleList([])
        self.e_conv.append(self.give_conv(3, 64))
        self.e_conv.append(self.give_conv(64, inner_channels))
        self.e_conv.append(self.give_conv(inner_channels, Num_channels, last=True))

        self.e_block = nn.ModuleList([self.res_block(inner_channels, inner_channels) for _ in range(7)])

        # Decoder
        self.d_conv = nn.ModuleList([])
        self.d_conv.append(self.give_conv(Num_channels, inner_channels, T=True))
        self.d_conv.append(self.give_conv(inner_channels, 64, T=True))
        self.d_conv.append(self.give_conv(64, 3, T=True, last=True))

        self.d_block = nn.ModuleList([self.res_block(inner_channels, inner_channels) for _ in range(7)])

        # Hasher
        self.hashed_layer = nn.Sequential(
            nn.BatchNorm1d(784),
            nn.Dropout(p=0.3),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.3),
            nn.Linear(100, K),
            nn.Tanh()
        )

    def encoder(self, x):
        conv = self.e_conv
        block = self.e_block

        c0 = conv[0](x)
        c1 = conv[1](c0)
        b0 = block[0](c1)
        b1 = block[1](b0 + c1)
        b2 = block[2](b0 + b1)
        sum1 = b2 + c1 + b1
        b3 = block[3](sum1)
        b4 = block[4](b3 + sum1)
        b5 = block[5](b3 + b4)
        sum2 = b5 + sum1 + b4
        b6 = block[6](sum2)
        sum3 = b6 + c1 + sum2
        c2 = conv[2](sum3)
        return torch.sigmoid(c2)

    def decoder(self, x):
        conv = self.d_conv
        block = self.d_block

        c0 = conv[0](x)
        b0 = block[0](c0)
        b1 = block[1](b0 + c0)
        b2 = block[2](b0 + b1)
        sum1 = b2 + c0 + b1
        b3 = block[3](sum1)
        b4 = block[4](b3 + sum1)
        b5 = block[5](b3 + b4)
        sum2 = b5 + sum1 + b4
        b6 = block[6](sum2)
        sum3 = b6 + c0 + sum2
        c1 = conv[1](sum3)
        c2 = conv[2](c1)
        return c2

    def hasher(self, x):
        to_hash = torch.median(x, dim=1, keepdim=True)
        out = self.hashed_layer(to_hash.view(-1, 784))
        return out

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        hashed = self.hasher(encoded)
        return encoded, decoded, hashed

    def give_conv(self, in_c, out_c, T=False, last=False):
        if T:
            conv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1)
        else:
            conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(5, 5), stride=(2, 2), padding=2)

        net = nn.Sequential(conv, nn.BatchNorm2d(out_c), nn.ReLU()) if last is False else nn.Sequential(conv, nn.BatchNorm2d(out_c))
        return net

    def res_block(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        net = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels)
        )
        return net
