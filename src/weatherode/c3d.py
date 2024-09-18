import torch
import torch.nn as nn
import torch.distributed as dist

class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
        )

class ResidualBlock2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, n_groups=1):
        super().__init__()
        mid_channels = (in_channels * out_channels * 3 * 3 * 3) // (in_channels * 3 * 3 + 3 * out_channels)
        self.conv1 = Conv2Plus1D(in_channels, out_channels, mid_channels)
        self.conv2 = Conv2Plus1D(out_channels, out_channels, mid_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.3)
        # self.attention = nn.MultiheadAttention(out_channels, num_heads)

        self.nan = nn.Identity()
        self.drop = nn.Dropout(0.1)

        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.norm1 = nn.GroupNorm(n_groups, out_channels) if norm else nn.Identity()
        self.norm2 = nn.GroupNorm(n_groups, out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer

        test = self.conv2(self.norm2(self.conv1(self.norm1(x.permute(1, 2, 0, 3, 4)))))
        if self._check_for_nan(test, "2Plus1D Conv"):
            h = self.activation(self.nan(self.conv1(self.norm1(x.permute(1, 2, 0, 3, 4)))))
            # Second convolution layer
            h = self.activation(self.nan(self.conv2(self.norm2(h))))
            h = self.drop(h)
        else:
            h = self.activation(self.bn1(self.conv1(self.norm1(x.permute(1, 2, 0, 3, 4)))))
            # Second convolution layer
            h = self.activation(self.bn2(self.conv2(self.norm2(h))))
            h = self.drop(h)

        if torch.isnan(self.bn1.running_mean).any() or torch.isnan(self.bn2.running_mean).any():
            print("NAN!!!!!!!!!\n\n\n")
            breakpoint()

        # Add the shortcut connection and return
        return (h + self.shortcut(x.permute(1, 2, 0, 3, 4))).permute(2, 0, 1, 3, 4)

    def _check_for_nan(self, tensor: torch.Tensor, step: str) -> bool:
        has_nan = torch.isnan(tensor).any().float()
        
        if dist.is_initialized():
            # 如果在分布式训练中，使用全局归约操作
            dist.all_reduce(has_nan, op=dist.ReduceOp.SUM)
        
        if has_nan > 0:
            if dist.is_initialized():
                rank = dist.get_rank()
                print(f"NaN detected on GPU {rank} at step: {step}")
                dist.barrier()  # 同步进程，让所有进程停留在相同的状态
            else:
                print(f"NaN detected in single-GPU training at step: {step}")
            return True
        return False


class ClimateResNet2Plus1D(nn.Module):
    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        cnn_layers = []

        self.residual_block_class = ResidualBlock2Plus1D
        self.inplanes = num_channels

        for idx in range(len(layers)):
            in_channels = num_channels if idx == 0 else hidden_size[idx - 1]
            out_channels = hidden_size[idx]
            cnn_layers.append(
                self.create_layer(
                    self.residual_block_class, in_channels, out_channels, layers[idx]
                )
            )

        self.cnn_layer_modules = nn.ModuleList(cnn_layers)

    def create_layer(self, block, in_channels, out_channels, reps):
        layers = []
        layers.append(block(in_channels, out_channels))
        for i in range(1, reps):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        output = data.float()

        for layer in self.cnn_layer_modules:
            output = layer(output)

        return output
