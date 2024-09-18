import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from weatherode.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

# from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from tqdm import tqdm
import warnings
import numpy as np
import torch.distributed as dist

class ViT(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        img_size=[32, 64],
        patch_size=2,
        depth=4,
        decoder_depth=2,
        embed_dim=1024,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):

        super().__init__()

        self.in_channel = in_channel

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channel, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim), requires_grad=True)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(img_size[0] / patch_size),
            int(img_size[1] / patch_size),
            cls_token=False,
        )

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channel * patch_size**2))
        self.head = nn.Sequential(*self.head)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        x = self.patch_embed(x)

        x += self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = self.head(x)

        x = x.reshape(x.shape[0], -1, H, W)

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.nan = nn.Identity()

        self.drop = nn.Dropout(0.1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm1 = nn.GroupNorm(n_groups, out_channels) if norm else nn.Identity()
        self.norm2 = nn.GroupNorm(n_groups, out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        test = self.conv2(self.norm2(self.conv1(self.norm1(x))))
        if self._check_for_nan(test, "Conv"):
            h = self.activation(self.nan(self.conv1(self.norm1(x))))
            # Second convolution layer
            h = self.activation(self.nan(self.conv2(self.norm2(h))))
            h = self.drop(h)
        else:
            h = self.activation(self.bn1(self.conv1(self.norm1(x))))
            # Second convolution layer
            h = self.activation(self.bn2(self.conv2(self.norm2(h))))
            h = self.drop(h)

        if torch.isnan(self.bn1.running_mean).any() or torch.isnan(self.bn2.running_mean).any():
            print("NAN!!!!!!!!!\n\n\n")
            breakpoint()

        # Add the shortcut connection and return
        return h + self.shortcut(x)

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

class ClimateResNet2D(nn.Module):
    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        cnn_layers = []

        self.residual_block_class = ResidualBlock
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


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, n_groups=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
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
        if self._check_for_nan(test, "3D Conv"):
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


class ClimateResNet3D(nn.Module):
    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        cnn_layers = []

        self.residual_block_class = ResidualBlock3D
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


class SelfAttentionConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionConv, self).__init__()
        self.query = self._conv_block(in_channels, in_channels // 8)
        self.key = self.key_conv_block(in_channels, in_channels // 8)
        self.value = self.key_conv_block(in_channels, out_channels)
        self.post_map = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding="same")
        self.out_ch = out_channels

    def _conv_block(self, n_in, n_out):
        return nn.Sequential(
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), padding="same"),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), padding="same"),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), padding="same")
        )
    
    def key_conv_block(self, n_in, n_out):
        return nn.Sequential(
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), padding="same"),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), padding="same"),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), padding="same")
        )
    
    def forward(self, x):
        size = x.size()
        x = x.float()
        q, k, v = self.query(x).flatten(-2, -1), self.key(x).flatten(-2, -1), self.value(x).flatten(-2, -1)
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_ch, size[-2], size[-1]).contiguous())
        return o


class OptimVelocity(nn.Module):
    def __init__(self, bs, num_vars, H, W):
        super(OptimVelocity, self).__init__()
        # [batch_size，变量数，H，W]
        self.v_x = torch.nn.Parameter(torch.randn(bs, num_vars, H, W))
        self.v_y = torch.nn.Parameter(torch.randn(bs, num_vars, H, W))

    def forward(self, data):
        # [batch_size, 5, 32, 64]
        u_y = torch.gradient(data, dim=2)[0] # (H,W) --> (y,x)
        u_x = torch.gradient(data, dim=3)[0]

        adv = self.v_x * u_x + self.v_y * u_y + data * (torch.gradient(self.v_y, dim=2)[0] + torch.gradient(self.v_x, dim=3)[0])
        return adv, self.v_x, self.v_y 


# def optimize_vel(x, prev_x, kernel):
#     with torch.enable_grad():
#         vel_model = OptimVelocity(*x.shape).to(x.device)
#         optimizer = torch.optim.Adam(vel_model.parameters(), lr=1)

#         # cubic estimate
#         prev_x = prev_x.view(prev_x.shape[0], prev_x.shape[1], -1)
#         t = torch.arange(3).flip(0).float().to(prev_x.device)

#         coeffs = natural_cubic_spline_coeffs(t, prev_x)
#         spline = NaturalCubicSpline(coeffs)
#         estimate_x = spline.derivative(t[-1]).view(-1, x.shape[1], x.shape[2], x.shape[3])
#         best_loss = float("inf")
#         for step in range(15):
#             optimizer.zero_grad()
#             out_x, v_x, v_y = vel_model(x)

#             kernel_v_x = v_x.view(v_x.shape[0], v_x.shape[1], -1, 1)
#             kernel_v_y = v_y.view(v_y.shape[0], v_y.shape[1], -1, 1)

#             kernel_expand = kernel.expand(v_x.shape[0], v_x.shape[1], kernel.shape[0], kernel.shape[1]).to(v_x.device)

#             v_x_kernel = torch.matmul(kernel_v_x.transpose(2, 3), kernel_expand)
#             final_x = torch.matmul(v_x_kernel, kernel_v_x).mean()
#             v_y_kernel = torch.matmul(kernel_v_y.transpose(2, 3), kernel_expand)
#             final_y = torch.matmul(v_y_kernel, kernel_v_y).mean()

#             vel_loss = torch.nn.MSELoss()(estimate_x.to(v_x.device), out_x.squeeze(1)) + 0.0000001 * (final_x + final_y)

#             if vel_loss.item() < best_loss:
#                 best_loss = vel_loss.item()
#                 final_vx = v_x
#                 final_vy = v_y

#             vel_loss.backward()
#             optimizer.step()
#     return final_vx, final_vy


class SparseLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        sparsity: sparsity of weight matrix
            Default: 0.9
        connectivity: user defined sparsity matrix
            Default: None
        small_world: boolean flag to generate small world sparsity
            Default: ``False``
        dynamic: boolean flag to dynamically change the network structure
            Default: ``False``
        deltaT (int): frequency for growing and pruning update step
            Default: 6000
        Tend (int): stopping time for growing and pruning algorithm update step
            Default: 150000
        alpha (float): f-decay parameter for cosine updates
            Default: 0.1
        max_size (int): maximum number of entries allowed before chunking occurrs
            Default: 1e8

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples:

        >>> m = nn.SparseLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        sparsity=0.9,
        connectivity=None,
        small_world=False,
        dynamic=False,
        deltaT=6000,
        Tend=150000,
        alpha=0.1,
        max_size=1e8,
    ):
        assert in_features < 2 ** 31 and out_features < 2 ** 31 and sparsity < 1.0
        assert (
            connectivity is None or not small_world
        ), "Cannot specify connectivity along with small world sparsity"
        if connectivity is not None:
            assert isinstance(connectivity, torch.LongTensor) or isinstance(
                connectivity, torch.cuda.LongTensor,
            ), "Connectivity must be a Long Tensor"
            assert (
                connectivity.shape[0] == 2 and connectivity.shape[1] > 0
            ), "Input shape for connectivity should be (2,nnz)"
            assert (
                connectivity.shape[1] <= in_features * out_features
            ), "Nnz can't be bigger than the weight matrix"
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.connectivity = connectivity
        self.small_world = small_world
        self.dynamic = dynamic
        self.max_size = max_size

        # Generate and coalesce indices : Faster to coalesce on GPU
        coalesce_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if not small_world:
            if connectivity is None:
                self.sparsity = sparsity
                nnz = round((1.0 - sparsity) * in_features * out_features)
                if in_features * out_features <= 10 ** 8:
                    indices = np.random.choice(
                        in_features * out_features, nnz, replace=False,
                    )
                    indices = torch.as_tensor(indices, device=coalesce_device)
                    row_ind = indices.floor_divide(in_features)
                    col_ind = indices.fmod(in_features)
                else:
                    warnings.warn(
                        "Matrix too large to sample non-zero indices without replacement, sparsity will be approximate",
                        RuntimeWarning,
                    )
                    row_ind = torch.randint(
                        0, out_features, (nnz,), device=coalesce_device,
                    )
                    col_ind = torch.randint(
                        0, in_features, (nnz,), device=coalesce_device,
                    )
                indices = torch.stack((row_ind, col_ind))
            else:
                # User defined sparsity
                nnz = connectivity.shape[1]
                self.sparsity = 1.0 - nnz / (out_features * in_features)
                connectivity = connectivity.to(device=coalesce_device)
                indices = connectivity
        else:
            # Generate small world sparsity
            self.sparsity = sparsity
            nnz = round((1.0 - sparsity) * in_features * out_features)
            assert nnz > min(
                in_features, out_features,
            ), "The matrix is too sparse for small-world algorithm; please decrease sparsity"
            offset = abs(out_features - in_features) / 2.0

            # Node labels
            inputs = torch.arange(
                1 + offset * (out_features > in_features),
                in_features + 1 + offset * (out_features > in_features),
                device=coalesce_device,
            )
            outputs = torch.arange(
                1 + offset * (out_features < in_features),
                out_features + 1 + offset * (out_features < in_features),
                device=coalesce_device,
            )

            # Creating chunks for small world algorithm
            total_data = in_features * out_features  # Total params
            chunks = math.ceil(total_data / self.max_size)
            split_div = max(in_features, out_features) // chunks  # Full chunks
            split_mod = max(in_features, out_features) % chunks  # Remaining chunk
            idx = (
                torch.repeat_interleave(torch.Tensor([split_div]), chunks)
                .int()
                .to(device=coalesce_device)
            )
            idx[:split_mod] += 1
            idx = torch.cumsum(idx, dim=0)
            idx = torch.cat([torch.LongTensor([0]).to(device=coalesce_device), idx])

            count = 0

            rows = torch.empty(0).long().to(device=coalesce_device)
            cols = torch.empty(0).long().to(device=coalesce_device)

            for i in range(chunks):
                inputs_ = (
                    inputs[idx[i] : idx[i + 1]]
                    if out_features <= in_features
                    else inputs
                )
                outputs_ = (
                    outputs[idx[i] : idx[i + 1]]
                    if out_features > in_features
                    else outputs
                )

                y = small_world_chunker(inputs_, outputs_, round(nnz / chunks))
                ref = torch.rand_like(y)

                # Refer to Eq.7 from Bipartite_small_world_network write-up
                mask = torch.empty(y.shape, dtype=bool).to(device=coalesce_device)
                mask[y < ref] = False
                mask[y >= ref] = True

                rows_, cols_ = mask.to_sparse().indices()

                rows = torch.cat([rows, rows_ + idx[i]])
                cols = torch.cat([cols, cols_])

            indices = torch.stack((cols, rows))
            nnz = indices.shape[1]

        values = torch.empty(nnz, device=coalesce_device)
        # indices, values = torch_sparse.coalesce(indices, values, out_features, in_features)

        self.register_buffer("indices", indices.cpu())
        self.weights = nn.Parameter(values.cpu())

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        if self.dynamic:
            self.deltaT = deltaT
            self.Tend = Tend
            self.alpha = alpha
            self.itr_count = 0

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.in_features ** 0.5
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        """ returns a torch.sparse.FloatTensor view of the underlying weight matrix
            This is only for inspection purposes and should not be modified or used in any autograd operations
        """
        weight = torch.sparse.FloatTensor(
            self.indices, self.weights, (self.out_features, self.in_features),
        )
        return weight.coalesce().detach()

    def forward(self, inputs):
        if self.training and self.dynamic:
            self.itr_count += 1
        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features

        # Handle dynamic sparsity
        if (
            self.training
            and self.dynamic
            and self.itr_count < self.Tend
            and self.itr_count % self.deltaT == 0
        ):
            # Drop criterion
            f_decay = (
                self.alpha * (1 + math.cos(self.itr_count * math.pi / self.Tend)) / 2
            )
            k = int(f_decay * (1 - self.sparsity) * self.weights.view(-1, 1).shape[0])
            n = self.weights.shape[0]

            neg_weights = -1 * torch.abs(self.weights)
            _, lm_indices = torch.topk(neg_weights, n - k, largest=False, sorted=False)

            self.indices = torch.index_select(self.indices, 1, lm_indices)
            self.weights = nn.Parameter(torch.index_select(self.weights, 0, lm_indices))

            device = inputs.device
            # Growth criterion
            new_weights = torch.zeros(k).to(device=device)
            self.weights = nn.Parameter(torch.cat((self.weights, new_weights), dim=0))

            new_indices = torch.zeros((2, k), dtype=torch.long).to(device=device)
            self.indices = torch.cat((self.indices, new_indices), dim=1)
            output = GrowConnections.apply(
                inputs,
                self.weights,
                k,
                self.indices,
                (self.out_features, self.in_features),
                self.max_size,
            )
        else:
            if len(output_shape) == 1:
                inputs = inputs.view(1, -1)
            inputs = inputs.flatten(end_dim=-2)

            # output = torch_sparse.spmm(self.indices, self.weights, self.out_features, self.in_features, inputs.t()).t()
            target = torch.sparse.FloatTensor(
                self.indices,
                self.weights,
                torch.Size([self.out_features, self.in_features]),
            ).to_dense()
            output = torch.mm(target, inputs.t()).t()

            if self.bias is not None:
                output += self.bias

        return output.view(output_shape)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, sparsity={}, connectivity={}, small_world={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.sparsity,
            self.connectivity,
            self.small_world,
        )



if __name__=='__main__':
    model = ResidualBlock3DWithAttention(3, 16, 4)

    input = torch.randn(10, 4, 3, 64, 64)

    model(input)

