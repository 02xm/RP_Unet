import torch
import torch.nn as nn

# determine a kind of up-sampling method [pixel_shuffle]
# replace this with original three insert
# first, [B, C, H, W] modify channels to s^2 multiple channels, then return to the original channels and
# convert to [B, C, SH, SW]
class Pixel_Shuffle1(nn.Module):
    def __init__(self, upscale_factor):
        super(Pixel_Shuffle1, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 64 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))  # 最终将输入转换成 [32, 9, H, W]
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 通过 Pixel Shuffle 来将 [32, 9, H, W] 重组为 [32, 1, 3H, 3W]

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x

class Pixel_Shuffle2(nn.Module):
    def __init__(self, upscale_factor):
        super(Pixel_Shuffle2, self).__init__()

        self.conv1 = nn.Conv2d(128, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 128 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))  # 最终将输入转换成 [32, 9, H, W]
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 通过 Pixel Shuffle 来将 [32, 9, H, W] 重组为 [32, 1, 3H, 3W]

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x

class Pixel_Shuffle3(nn.Module):
    def __init__(self, upscale_factor):
        super(Pixel_Shuffle3, self).__init__()

        self.conv1 = nn.Conv2d(256, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 256 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))  # 最终将输入转换成 [32, 9, H, W]
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 通过 Pixel Shuffle 来将 [32, 9, H, W] 重组为 [32, 1, 3H, 3W]

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x

class Pixel_Shuffle4(nn.Module):
    def __init__(self, upscale_factor):
        super(Pixel_Shuffle4, self).__init__()

        self.conv1 = nn.Conv2d(512, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 512 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))  # 最终将输入转换成 [32, 9, H, W]
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 通过 Pixel Shuffle 来将 [32, 9, H, W] 重组为 [32, 1, 3H, 3W]

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


