import torch
import torch.nn as nn
import torch.nn.functional as F
import natten as nt
from torchvision import transforms


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super(ConvNeXtBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2,
                                        groups=dim)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.norm = nn.LayerNorm(4 * dim)
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pointwise_conv1(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.pointwise_conv2(x)
        x = x.permute(0, 3, 1, 2)
        x += shortcut
        return x


class ConvNeXt_T(nn.Module):
    def __init__(self):
        super(ConvNeXt_T, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 224, kernel_size=(4, 4), stride=(4, 4)),
            *[ConvNeXtBlock(224) for _ in range(3)],
        )

    def forward(self, x):
        x = self.features(x)
        return x


class TinyResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TinyResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class TinyResNetX3(nn.Module):
    def __init__(self):
        super(TinyResNetX3, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(out_channels=64, blocks=3, stride=1)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        strides = [stride] + [1] * (blocks - 1)  # Only the first block could have stride != 1
        for stride in strides:
            layers.append(TinyResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return x


class FeatureExtractionModule(nn.Module):
    def __init__(self, padding_3x3=1, padding_5x5=2):
        super(FeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), padding=padding_3x3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), padding=padding_5x5)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=padding_3x3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class StackedBiNN(nn.Module):
    def __init__(self):
        super(StackedBiNN, self).__init__()
        self.feature_extract = FeatureExtractionModule()
        self.layer1 = TinyResNetX3()
        self.layer2 = ConvNeXt_T()
        self.down_sample = nn.MaxPool2d(kernel_size=(4, 4))
        # self.channel_reduce1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1))
        self.channel_reduce = nn.Conv2d(in_channels=224, out_channels=64, kernel_size=(1, 1))
        self.attention_transformer = nt.NeighborhoodAttention2D(dim=64, num_heads=4, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 56 * 56, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.feature_extract(x)
        out1 = self.layer1(x)
        out2 = self.layer2(x)

        # print(f"Layer1 output size{out1.size()}")
        # print(f"Layer2 output size{out2.size()}")
        print(out1.permute(0, 3, 2, 1).size())

        out2_channel_reduced = self.channel_reduce(out2)
        # print(f"Layer2 Channel Reduce size: {out2_channel_reduced.size()}")
        out1_down_sample = self.down_sample(out1)
        # print(f"Layer1 Down Sample size: {out1_down_sample.size()}")
        first_concat = torch.cat((out1_down_sample, out2_channel_reduced), dim=1)
        # print(f"First Concat size: {first_concat.size()}")
        neighbour_attention_out = self.attention_transformer(out1_down_sample.permute(0, 3, 2, 1))
        # print(f"Attention size: {neighbour_attention_out.size()}")
        neighbour_attention_out = neighbour_attention_out.permute(0, 3, 1, 2)
        second_concat = torch.cat((neighbour_attention_out, self.channel_reduce(out2)), dim=1)
        # print(f"Second Concat size: {second_concat.size()}")
        final_fusion = torch.cat((first_concat, second_concat), dim=1)
        # print(f"Last Fusion size: {final_fusion.size()}")
        cat = self.flatten(final_fusion)
        return self.fc(cat)


if __name__ == '__main__':
    sbnn = StackedBiNN()
    # print(sbnn)
    input_data = torch.randn(1, 3, 224, 224)
    print(sbnn(input_data).size())
    # out1, out2, out3 = sbnn(input_data)
    # print(out1.size(), out2.size(), out3.size())
