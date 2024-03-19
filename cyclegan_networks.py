import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.models import resnet18

# from einops.layers.torch import Rearrange

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


class EnhancedPatchGANDiscriminator32(nn.Module):
    def __init__(self, input_nc):
        super(EnhancedPatchGANDiscriminator32, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=1, padding=1, normalization=True,
                                instance_norm=True, use_spectral_norm=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if use_spectral_norm:
                layers = [spectral_norm(layer) for layer in layers]
            if normalization:
                if instance_norm:
                    layers.append(nn.InstanceNorm2d(out_filters))
                else:
                    layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Adding more layers and increasing the capacity
        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalization=False, stride=2, use_spectral_norm=True),
            # 128x128 -> 64x64
            *discriminator_block(64, 128, stride=2, use_spectral_norm=True),  # 64x64 -> 32x32
            *discriminator_block(128, 256, stride=2, use_spectral_norm=True),  # 32x32 -> 16x16
            *discriminator_block(256, 512, stride=2, use_spectral_norm=True),  # 16x16 -> 8x8
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 8x8 -> 5x5
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


class CombinedDiscriminator(nn.Module):
    def __init__(self, B=10, C=128, patch_size=32, num_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.T = nn.Parameter(torch.randn(input_features, output_features))

        # PatchGAN components
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, C, kernel_size=4, stride=1, padding=1)

        # Minibatch Discrimination component
        self.T = nn.Parameter(torch.randn(C, B))

    def forward(self, x):
        # x shape: (batch_size, num_channels, height, width)

        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, self.num_channels, self.patch_size, self.patch_size)

        # Process each patch with PatchGAN layers
        x = F.leaky_relu(self.conv1(patches), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.conv3(x)  # Now x is (N, C, H, W), where N is the total number of patches

        # Flatten output for minibatch discrimination
        N, C, H, W = x.size()
        x = x.view(N, -1)  # Now x is (N, C*H*W)

        # Minibatch discrimination
        M = x.mm(self.T)  # (N, B)
        M = M.unsqueeze(0) - M.unsqueeze(1)  # Broadcasting to create pairwise differences
        M = torch.exp(-torch.abs(M).sum(2))  # (N, N)
        o = M.sum(0).view(-1, 1)  # Sum over rows for each sample, reshape to (N, 1)

        # Concatenate patch output and minibatch discrimination output
        x = torch.cat((x, o), 1)  # Now x is (N, C*H*W + 1)

        # Note: This output is per patch. To get per image, you'd need to reshape and aggregate appropriately
        return torch.sigmoid(x)


"""PATCHGAN VISALOIZATION"""


def visualize_realness_map(image, scores, image_size=256, patch_size=32):
    """
    Visualizes a realness map as a heatmap overlaid on the original image.

    :param image: The input image as a PIL image or a tensor.
    :param scores: The discriminator's output scores as a 2D tensor.
    :param image_size: The size of the image (assumed to be square).
    :param patch_size: The size of each patch that was assessed by the discriminator.
    """
    # Normalize and resize the scores to match the image size
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores_upsampled = torch.nn.functional.interpolate(scores.unsqueeze(0).unsqueeze(0),
                                                       size=image_size,
                                                       mode='bilinear',
                                                       align_corners=False).squeeze()

    # Convert the image to a tensor if it is a PIL image
    if not isinstance(image, torch.Tensor):
        transform = transforms.ToTensor()
        image = transform(image)

    # Ensure the image is between 0 and 1
    image = (image - image.min()) / (image.max() - image.min())

    # Convert the image to numpy for plotting
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np, interpolation='nearest')
    plt.imshow(scores_upsampled.cpu().numpy(), cmap='jet', alpha=0.5, interpolation='nearest')
    plt.colorbar()
    plt.title('Realness Map Overlay')
    plt.show()


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


class DoubleConvIN(nn.Module):
    """(convolution => [IN] => LeakyReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
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


class OutConvTanh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvTanh, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)  # Ensures output is in the range of -1 to 1
        return x


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
        self.outc = OutConvTanh(64, n_classes)

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




""""""""""""


class TransformerBlock(nn.Module):
    # Assuming this is a simplified Transformer block for illustration
    def __init__(self, embed_dim, depth, heads, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Apply layer norm followed by multi-head attention and a feed-forward network
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, heads, mlp_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UNet2ConvTrans(nn.Module):
    def __init__(self, n_channels, n_classes, num_transformer_blocks=1, depth=1, heads=8, mlp_dim=2048, dropout=0.1):
        super(UNet2ConvTrans, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # We initialize TransformerBlocks with correct feature size (512)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(512, depth, heads, mlp_dim, dropout) for _ in range(num_transformer_blocks)]
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

        # Correctly reshape x5 for transformer blocks: Flatten spatial dimensions
        x5 = x5.flatten(2).transpose(1, 2)  # Now x5 should have shape [batch, seq_len, channels]

        # Pass through each transformer block
        for block in self.transformer_blocks:
            x5 = block(x5)

        # Reshape back to feature map format
        x5 = x5.transpose(1, 2).view(-1, 512, 8, 8)  # Ensure this matches the expected output shape

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet2ConvTrans2(nn.Module):
    def __init__(self, n_channels, n_classes, num_transformer_blocks=1, heads=8, mlp_dim=2048, dropout=0.1):
        super(UNet2ConvTrans2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Initialize the specified number of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(512, heads, mlp_dim, dropout) for _ in range(num_transformer_blocks)
        ])

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

        # Reshape x5 to be compatible with the transformer
        x5 = x5.flatten(2).transpose(1, 2)  # Change shape to [batch, seq_len, channels]

        # Pass x5 through each transformer block
        for block in self.transformer_blocks:
            x5 = block(x5)

        # Reshape x5 back to feature map format before upsampling
        x5 = x5.transpose(1, 2).view(-1, 512, 8, 8)  # Adjust shape based on your feature map size

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class PixelwiseViT(nn.Module):

    def __init__(
            self, features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm, image_shape, rezero=True, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, C, H * W)
        itokens = x.view(*x.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        itokens = itokens.permute((0, 2, 1))

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W)
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W)
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        return result


class UNet2ConvViT(nn.Module):
    def __init__(self, n_channels, n_classes, features, n_heads, n_blocks, ffn_features, embed_features, activ, norm,
                 rezero=True):
        super(UNet2ConvViT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Define the encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Define the Vision Transformer module at the bottleneck
        self.vit_bottleneck = PixelwiseViT(
            features=features,
            n_heads=n_heads,
            n_blocks=n_blocks,
            ffn_features=ffn_features,
            embed_features=embed_features,
            activ=activ,
            norm=norm,
            image_shape=(512, 8, 8),  # Assuming the bottleneck feature size is (512, 8, 8)
            rezero=rezero
        )

        # Define the decoder (upsampling path)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConvTanh(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply the Vision Transformer at the bottleneck
        x5 = self.vit_bottleneck(x5)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet2ConvResLarge(nn.Module):
    def __init__(self, n_channels, n_classes, num_residual_blocks=2):
        super(UNet2ConvResLarge, self).__init__()
        # Adjusting the number of filters and depths accordingly
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Increased filter count
        self.down5 = Down(1024, 1024)  # Extra downscaling for larger input

        # Residual blocks at the lowest resolution
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(1024) for _ in range(num_residual_blocks)]
        )

        # Upscaling back to the original size
        self.up1 = Up(2048, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.up5 = Up(128, 64)  # Additional upscaling layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x6 = self.residual_blocks(x6)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits


class AOTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(AOTBlock, self).__init__()
        self.branches = nn.ModuleList()
        branch_out_channels = int(out_channels / len(dilation_rates))

        for rate in dilation_rates:
            self.branches.append(
                nn.Conv2d(in_channels, branch_out_channels, kernel_size=3, padding=rate, dilation=rate))

        self.merge = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        merged = self.merge(torch.cat(branch_outputs, 1))
        return merged


class UNet2ConvResAOT(nn.Module):
    def __init__(self, n_channels, n_classes, num_aot_blocks=2):
        super(UNet2ConvResAOT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Assuming Down, Up, DoubleConv, and OutConvTanh are defined elsewhere in your code
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Replace residual blocks with AOT blocks
        self.aot_blocks = nn.Sequential(
            *[AOTBlock(512, 512, [1, 2, 4, 8]) for _ in range(num_aot_blocks)]
        )

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConvTanh(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Passing through AOT blocks
        x5 = self.aot_blocks(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


""""""""


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, stride=1, padding=1):
        super(ModulatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.style_fc = nn.Linear(style_dim, out_channels)

    def forward(self, x, style):
        batch_size, _, _, _ = x.shape
        style = self.style_fc(style).view(batch_size, -1, 1, 1)
        weights = self.conv.weight.unsqueeze(0) * style.unsqueeze(2)

        output = torch.empty(batch_size, self.conv.out_channels, x.shape[2], x.shape[3], device=x.device)
        for i in range(batch_size):
            output[i] = F.conv2d(x[i:i + 1], weights[i], bias=None, stride=self.conv.stride, padding=self.conv.padding)
        return output


class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        return enc1, enc2, enc3


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, style_dim):
        super(UNetDecoder, self).__init__()
        self.modconv1 = ModulatedConv2d(256 + 256, 128, 3, style_dim)  # Features from enc3 concatenated
        self.modconv2 = ModulatedConv2d(128 + 128, 64, 3, style_dim)  # Features from enc2 concatenated
        self.final_conv = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                        nn.Conv2d(64, out_channels, 3, padding=1))

    def forward(self, x, enc1, enc2, enc3, style):

        x = torch.cat([x, enc3], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.modconv1(x, style))

        x = torch.cat([x, enc2], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.modconv2(x, style))

        x = torch.cat([x, enc1], dim=1)
        # x = F.interpolate(x, scale_factor=2)
        x = self.final_conv(x)
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


class UNetGeneratorWithTransformer(nn.Module):
    def __init__(self):
        super(UNetGeneratorWithTransformer, self).__init__()
        self.encoder = UNetEncoder(1)
        self.transformer = TransformerBottleneck(num_tokens=1024, emb_dim=256, num_heads=8, depth=6, style_dim=128)
        self.decoder = UNetDecoder(1, 128)

    def forward(self, x):
        enc1, enc2, enc3 = self.encoder(x)

        # Flatten and permute for transformer, assuming enc3 is used for style generation
        b, c, h, w = enc3.shape
        enc3_flat = enc3.view(b, c, h * w).permute(0, 2, 1)
        style = self.transformer(enc3_flat)

        x = self.decoder(enc3, enc1, enc2, enc3, style)
        return x
