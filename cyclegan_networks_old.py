import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x, skip_input=None):
        x = self.model(x)
        if self.dropout:
            x = self.dropout(x)
        # Concatenate only if skip_input is provided
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(GeneratorUNet, self).__init__()

        # Downsample
        self.down1 = UNetDown(input_nc, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.25)
        self.down4 = UNetDown(256, 512, dropout=0.25)
        self.down5 = UNetDown(512, 512, dropout=0.5)

        # Upsample
        self.up1 = UNetUp(512, 512, dropout=0.25)
        self.up2 = UNetUp(1024, 256, dropout=0.5)
        self.up3 = UNetUp(512, 128, dropout=0.25)
        self.up4 = UNetUp(256, 64, dropout=0.25)
        self.up5 = UNetUp(128, 64)  # Additional upsampling layer

        # Final Convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, output_nc, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Upsample and skip connections
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4)  # Additional upsampling step

        return self.final(u5)


class GeneratorUNetResNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_res_blocks=2):
        super(GeneratorUNetResNet, self).__init__()

        # Downsample
        self.down1 = UNetDown(input_nc, 64, normalize=False)
        self.down2 = UNetDown(64, 128, dropout=0.1)
        self.down3 = UNetDown(128, 256, dropout=0.1)
        self.down4 = UNetDown(256, 512, dropout=0.1)
        self.down5 = UNetDown(512, 512, dropout=0.3)

        # ResNet blocks in the middle
        self.res_blocks = nn.Sequential(*[ResnetBlock(512, use_dropout=True) for _ in range(num_res_blocks)])

        # Upsample
        self.up1 = UNetUp(512, 512, dropout=0.3)
        self.up2 = UNetUp(1024, 256, dropout=0.1)
        self.up3 = UNetUp(512, 128, dropout=0.1)
        self.up4 = UNetUp(256, 64, dropout=0.1)
        self.up5 = UNetUp(128, 64)

        # Final Convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, output_nc, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Pass through ResNet blocks
        res_out = self.res_blocks(d5)

        # Upsample and skip connections
        u1 = self.up1(res_out, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4)

        return self.final(u5)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GeneratorUNetResNet2(nn.Module):
    def __init__(self, input_nc, output_nc, num_res_blocks=2, bilinear=True):
        super(GeneratorUNetResNet2, self).__init__()

        # Downsample
        self.down1 = Down(input_nc, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # ResNet blocks in the middle
        self.res_blocks = nn.Sequential(*[ResnetBlock(512, use_dropout=True) for _ in range(num_res_blocks)])

        # Upsample
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 64, bilinear)

        # Final Convolution
        self.out_conv = OutConv(64, output_nc)

    def forward(self, x):
        # Downsample
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Pass through ResNet blocks
        x_res = self.res_blocks(x4)

        # Upsample and concatenate skip connections
        x = self.up1(x_res, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)

        # Final output
        output = self.out_conv(x)
        return output

"""ADVANCED VISION TRANSFROMERS"""


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, stride=1, padding=1):
        super(ModulatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.style_fc = nn.Linear(style_dim, out_channels)

    def forward(self, x, style):
        # Modulate convolution weights based on style
        batch_size, _, _, _ = x.shape
        style = self.style_fc(style).view(batch_size, -1, 1, 1)
        weights = self.conv.weight.unsqueeze(0) * style.unsqueeze(2)

        # Initialize an output tensor.
        output = torch.empty(batch_size, self.conv.out_channels, x.shape[2], x.shape[3], device=x.device)

        # Apply convolutions individually.
        for i in range(batch_size):
            output[i] = F.conv2d(x[i:i + 1], weights[i], bias=None, stride=self.conv.stride, padding=self.conv.padding)

        return output


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.resnet = nn.Sequential(*list(resnet18(weights=False).children())[1:-2])

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resnet(x)
        return x


class TransformerBottleneck(nn.Module):
    def __init__(self, num_tokens, emb_dim, num_heads, depth, style_dim):
        super(TransformerBottleneck, self).__init__()
        self.token_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, emb_dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.TransformerEncoderLayer(emb_dim, num_heads))
        self.to_style = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, style_dim)
        )

    def forward(self, x):
        b, _, _ = x.shape
        x = x + self.token_embedding[:, 1:, :]
        cls_token = self.token_embedding[:, 0, :].unsqueeze(1).repeat(b, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        style = self.to_style(x[:, 0, :])
        return style


class Decoder(nn.Module):
    def __init__(self, out_channels, style_dim):
        super(Decoder, self).__init__()
        self.modconv1 = ModulatedConv2d(512, 256, 3, style_dim)
        self.modconv2 = ModulatedConv2d(256, 128, 3, style_dim)
        self.modconv3 = ModulatedConv2d(128, 64, 3, style_dim)  # Define third modconv layer.
        self.modconv4 = ModulatedConv2d(64, 64, 3, style_dim)  # Define fourth modconv layer.
        self.modconv5 = ModulatedConv2d(64, 64, 3, style_dim)  # Define fourth modconv layer.
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, style):
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.modconv1(x, style))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.modconv2(x, style))
        x = F.interpolate(x, scale_factor=2)  # Upscale the spatial dimensions.
        x = F.relu(self.modconv3(x, style))  # Apply modulated convolution.
        x = F.interpolate(x, scale_factor=2)  # Upscale again.
        x = F.relu(self.modconv4(x, style))  # Apply another modulated convolution.
        x = F.interpolate(x, scale_factor=2)  # Upscale again.
        x = F.relu(self.modconv5(x, style))  # Apply another modulated convolution.
        x = self.final_conv(x)
        return x


class UNetGeneratorWithTransformer(nn.Module):
    def __init__(self):
        super(UNetGeneratorWithTransformer, self).__init__()
        self.encoder = Encoder(1)  # Grayscale input
        self.transformer = TransformerBottleneck(num_tokens=16, emb_dim=512, num_heads=8, depth=6, style_dim=128)  # num_tokens=256
        self.decoder = Decoder(1, 128)  # Grayscale output

    def forward(self, x):
        x = self.encoder(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        style = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = self.decoder(x, style)
        return x
