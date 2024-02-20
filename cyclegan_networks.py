import torch
import torch.nn as nn
import torch.nn.functional as F

"""DISCRIMINATORS"""


class Discriminator(nn.Module):
    def __init__(self, input_nc, image_size):
        super(Discriminator, self).__init__()

        flattened_size = 128 * (128 // 2 // 2) * (128 // 2 // 2)

        # Define the sequence of convolutional layers
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)

        return x


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, instance_norm=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                if instance_norm:
                    layers.append(nn.InstanceNorm2d(out_filters))
                else:
                    layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        """Returns discriminator score for each patch of the input image"""
        img_x = self.model(img)
        return torch.sigmoid(img_x)


class PatchGANDiscriminator32(nn.Module):
    def __init__(self, input_nc):
        super(PatchGANDiscriminator32, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=1, padding=1, normalization=True,
                                instance_norm=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if normalization:
                if instance_norm:
                    layers.append(nn.InstanceNorm2d(out_filters))
                else:
                    layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Adjusting blocks to reduce downsampling and aim for 32x32 patches
        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalization=False, stride=2),  # 128x128 -> 64x64
            *discriminator_block(64, 128, stride=2),  # 64x64 -> 32x32
            nn.Conv2d(128, 1, kernel_size=4, padding=1)  # Final layer to predict on 32x32 patches
        )

    def forward(self, img):
        """Returns discriminator score for each patch of the input image"""
        img_x = self.model(img)
        return torch.sigmoid(img_x)


class MiniBatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super(MiniBatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x shape: [batch, in_features]
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # add dummy batch dimension for broadcasting
        M_T = M.permute(1, 0, 2, 3)
        norm = torch.abs(M - M_T).sum(3)  # L1 norm
        expnorm = torch.exp(-norm)
        o_b = expnorm.sum(0)  # sum over batches
        if self.mean:
            o_b /= x.size(0)

        x = torch.cat([x, o_b], 1)
        return x


class MBDiscriminator(nn.Module):
    def __init__(self, input_nc, image_size):
        super(MBDiscriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flattened_size = 128 * (image_size // 2 // 2) * (image_size // 2 // 2)

        # Mini-Batch Discrimination
        self.mb_discrim = MiniBatchDiscrimination(self.flattened_size, 100, 5)

        # Linear Layer
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size + 100, 1),  # Adjust input size to account for mini-batch features
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.mb_discrim(x)  # Mini-Batch Discrimination
        x = self.linear_layers(x)
        return x


"""GENERATORS"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2Conv(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2Conv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Element-wise addition
        out = self.relu(out)
        return out


class UNet2ConvRes(nn.Module):
    def __init__(self, n_channels, n_classes, num_residual_blocks=2):
        super(UNet2ConvRes, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Add residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(512) for _ in range(num_residual_blocks)]
        )

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Passing through residual blocks
        x5 = self.residual_blocks(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
