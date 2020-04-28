import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

bottleneck_dim = 2048


class ResNet_AE(nn.Module):
    def __init__(self, K):
        super(ResNet_AE, self).__init__()

        # encoding components
        self.resnet = models.resnet50(pretrained=True)
        # modules = list(resnet.children())[:-2]      # delete the last fc,avgpool layer.
        # self.resnet = nn.Sequential(*modules)
        self.resnet.fc = nn.Linear(2048, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, 2048)

        # Decoder
        self.d_up_1 = self.give_upconv(2048, 1024)
        self.d_up_2 = self.give_upconv(1024, 512)
        self.d_up_3 = self.give_upconv(512, 128)
        self.d_up_4 = self.give_upconv(128, 32)
        self.d_up_5 = self.give_upconv(32, 3)

        self.d_conv_1 = self.res_block(1024, 1024)
        self.d_conv_2 = self.res_block(1024, 1024)
        self.d_conv_3 = self.res_block(512, 512)
        self.d_conv_4 = self.res_block(512, 512)
        self.d_conv_5 = self.res_block(128, 128)
        self.d_conv_6 = self.res_block(128, 128)
        self.d_conv_16 = self.res_block(128, 128)

        # HASHING LAYER
        self.hashed_layer = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim),
            # nn.Dropout(p=0.2),
            nn.Linear(bottleneck_dim, K),
            nn.ReLU(),
            nn.BatchNorm1d(K),
            # nn.Dropout(p=0.2),
            nn.Linear(K, K),
            nn.Tanh()
        )

    def give_upconv(self, in_c, out_c):
        net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
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

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        return x

    def decode(self, z):
        z = self.from_bottleneck(z)
        z = F.interpolate(z.view(-1, 2048, 1, 1), scale_factor=7)
        # print(z.size())
        dc1 = self.d_up_1(z)
        dblock1 = self.d_conv_1(dc1)
        dblock2 = self.d_conv_2(dblock1 + dc1)
        dsum1 = dc1 + dblock2
        dc2 = self.d_up_2(dsum1)
        dblock3 = self.d_conv_3(dc2)
        dblock4 = self.d_conv_4(dblock3 + dc2)
        dsum2 = dc2 + dblock4
        dc3 = self.d_up_3(dsum2)
        dblock5 = self.d_conv_5(dc3)
        dblock6 = self.d_conv_6(dblock5 + dc3)
        dsum3 = dc3 + dblock6
        dblock16 = self.d_conv_16(dsum3)
        dsum6 = dblock16 + dc3 + dsum3
        dc4 = self.d_up_4(dsum6)
        dc5 = self.d_up_5(dc4)
        return dc5

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)
        hashed = self.hashed_layer(z)
        return x_reconst, hashed


if __name__ == '__main__':
    net = ResNet_AE(K=16).cuda()
    net = nn.DataParallel(net).cuda()
    for name, _ in net.named_parameters():
        print(name)
