import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dtype=None, device=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, dtype=dtype, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False, dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, dtype=dtype, device=device)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                          bias=False, dtype=dtype, device=device),
                nn.BatchNorm2d(out_channels, dtype=dtype, device=device)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, cifar=True, madam=False, dtype=None, device=None):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # maxpool is disabled for CIFAR10/100 datasets as input images are small (32x32)
        if cifar:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False, dtype=dtype, device=device)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, dtype=dtype, device=device)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64, dtype=dtype, device=device)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dtype=dtype, device=device)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dtype=dtype, device=device)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dtype=dtype, device=device)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dtype=dtype, device=device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, dtype=dtype, device=device)

        if madam:
            self._initialize_weights_madam()
        else:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, dtype=None, device=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dtype=dtype, device=device))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)

        return nn.functional.log_softmax(out, dim=1)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def _initialize_weights_madam(self, residual_scale=0.1, bias_scale=0.01):

        @torch.no_grad()
        def signed_constant_(tensor, magnitude):
            signs = torch.empty_like(tensor).uniform_(-1.0, 1.0)
            signs = torch.where(signs < 0, -1.0, 1.0)
            tensor.copy_(signs * magnitude)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    signed_constant_(m.bias, bias_scale)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    signed_constant_(m.bias, bias_scale)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    signed_constant_(m.bias, bias_scale)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, residual_scale)
