import torch
import torch.nn as nn
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_embed_dim: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding="same"
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding="same"
        )
        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding="same"
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.drop = nn.Dropout(0.1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm1 = nn.GroupNorm(n_groups, out_channels) if norm else nn.Identity()
        self.norm2 = nn.GroupNorm(n_groups, out_channels) if norm else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, 6 * out_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        h = self.activation(self.bn1(self.conv1(self.norm1(x))))
        h = modulate(h, shift1, scale1)
        # First convolution layer
        h = self.activation(self.bn1(self.conv2(h)))
        h = h * gate1.unsqueeze(2).unsqueeze(3)

        # Second convolution layer
        h = modulate(self.norm2(h), shift2, scale2)
        h = self.activation(self.bn3(self.conv3(h)))
        h = h * gate2.unsqueeze(2).unsqueeze(3)
        
        h = self.drop(h)
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class ClimateResNet2DTime(nn.Module):
    def __init__(self, num_channels, layers, hidden_size, t_embed_dim=256):
        super().__init__()
        cnn_layers = []

        self.residual_block_class = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            in_channels = num_channels if idx == 0 else hidden_size[idx - 1]
            out_channels = hidden_size[idx]
            cnn_layers.append(
                self.create_layer(
                    self.residual_block_class, in_channels, out_channels, layers[idx], t_embed_dim
                )
            )

        self.cnn_layer_modules = nn.ModuleList(cnn_layers)
        self.t_embedder = TimestepEmbedder(hidden_size[0], t_embed_dim)

    def create_layer(self, block, in_channels, out_channels, reps, t_embed_dim):
        layers = []
        layers.append(block(in_channels, out_channels, t_embed_dim=t_embed_dim))
        for i in range(1, reps):
            layers.append(block(out_channels, out_channels, t_embed_dim=t_embed_dim))

        return nn.Sequential(*layers)

    def forward(self, data, t):
        output = data.float()
        t_emb = self.t_embedder(t)

        for layer in self.cnn_layer_modules:
            for block in layer:
                output = block(output, t_emb)

        return output
