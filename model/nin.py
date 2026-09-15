import torch.nn as nn
import torch


class MLPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        out_channels3,
        kernel_size,
        padding,
    ):
        super().__init__()

        self.mlpconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels1, out_channels2, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels2, out_channels3, 1),
        )

    def forward(self, x):
        return self.mlpconv(x)


class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.mlpconv1 = MLPConv(
            in_channels=3,
            out_channels1=64,
            out_channels2=64,
            out_channels3=64,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.mlpconv2 = MLPConv(
            in_channels=64,
            out_channels1=128,
            out_channels2=128,
            out_channels3=128,
            kernel_size=3,
            padding=1,
        )

        self.mlpconv3 = MLPConv(
            in_channels=128,
            out_channels1=128,
            out_channels2=64,
            out_channels3=num_classes,
            kernel_size=3,
            padding=1,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.mlpconv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.mlpconv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.mlpconv3(x)
        x = self.pool(x)
        x = self.gap(x)

        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = NIN()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
