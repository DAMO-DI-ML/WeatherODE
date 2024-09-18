# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from weatherode.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from weatherode.ode_utils import ClimateResNet2D, ClimateResNet3D, SelfAttentionConv, ViT, SparseLinear
from weatherode.dit import DiT
from weatherode.cnn_dit import ClimateResNet2DTime
from weatherode.c3d import ClimateResNet2Plus1D
from torchdiffeq import odeint
import torch.distributed as dist

class WeatherODE(nn.Module):
    """Implements the WeatherODE model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    """

    def __init__(
        self,
        default_vars,
        method,
        img_size=[32, 64],
        patch_size=2,
        layers=[5, 5, 3, 2], # [5, 3, 2],
        hidden=[512, 128, 64], #[256, 64],
        depth=4,
        use_err=True,
        err_type="2D",
        err_with_x=False,
        err_with_v=False,
        err_with_std=False,
        drop_rate=0.1,
        time_steps=12,
        time_interval=0.001,
        rtol=1e-9,
        atol=1e-11,
        predict_list=[6],
        gradient_loss=False
    ):
        super().__init__()

        self.default_vars = default_vars
        self.method = method
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.time_interval = time_interval
        self.rtol = rtol
        self.atol = atol
        
        self.layers = layers
        self.hidden = hidden + [2 * len(self.default_vars)]

        self.use_err = use_err
        self.drop_rate = drop_rate
        self.predict_list = predict_list
        self.gradient_loss = gradient_loss
        self.err_with_x = err_with_x
        self.err_with_v = err_with_v
        self.err_type = err_type

        self.v_net = ClimateResNet2D(3 * len(self.default_vars), self.layers, self.hidden)
        # self.v_net = ViT(3 * len(self.default_vars), 2 * len(self.default_vars))

        # t(1), day_t(2), sea_t(2), x(5), nabla_x(10), v(10), lat_lon(2), pos(6), pos_time(24)
        input_channels = 37 + len(self.default_vars) * 5

        self.model = ViT(input_channels, 2 * len(self.default_vars), depth=depth, patch_size=patch_size, img_size=img_size)
        # self.model = ClimateResNet2D(input_channels, self.layers, self.hidden)

        self.linear_model = SparseLinear(input_channels, 2 * len(self.default_vars))
        # ClimateResNet2D(input_channels, self.layers, self.hidden)

        # x(5), nabla_x(10), v(10), lat_lon(2), pos(6)
        if self.use_err:
            noise_input_channels = 8 + len(self.default_vars)
            noise_input_channels = noise_input_channels + 2 * len(self.default_vars) if self.err_with_v else noise_input_channels
            noise_input_channels = noise_input_channels + len(self.default_vars) if self.err_with_x else noise_input_channels

            noise_hidden = hidden + [len(self.default_vars)]

            if err_type == "vit":
                self.noise_model =  ViT(noise_input_channels, len(self.default_vars), patch_size=patch_size, img_size=img_size)
            elif err_type == "2D":
                self.noise_model = ClimateResNet2D(noise_input_channels, self.layers, noise_hidden) 
            elif err_type == "3D":
                self.noise_model = ClimateResNet3D(noise_input_channels, self.layers, noise_hidden)
            elif err_type == "2+1D":
                self.noise_model = ClimateResNet2Plus1D(noise_input_channels, self.layers, noise_hidden)
            elif err_type == "DiT":
                self.noise_model = DiT(input_size=tuple(img_size), in_channels=noise_input_channels, out_channels=len(self.default_vars), depth=4, hidden_size=768, patch_size=2, num_heads=12)
            elif err_type == "2DTime":
                self.noise_model = ClimateResNet2DTime(noise_input_channels, self.layers, noise_hidden, t_embed_dim=hidden[0])

        self.var_map = self.create_var_map()

    def create_var_map(self):
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def pde(self, t, x):
        vx = x[:, len(self.default_vars) : 2 * len(self.default_vars)]
        vy = x[:, 2 * len(self.default_vars) : 3 * len(self.default_vars)]

        v = torch.cat([vx, vy], 1)

        new_lat_lon = x[:, 3 * len(self.default_vars): 3 * len(self.default_vars) + 2]

        pos_feats = x[:, 3 * len(self.default_vars) + 2:]

        x = x[:, : len(self.default_vars)]

        x_grad_x = torch.gradient(x, dim=3)[0]
        x_grad_y = torch.gradient(x, dim=2)[0]

        nabla_x = torch.cat([x_grad_x, x_grad_y], 1)

        t_emb = ((t * (1 / self.time_interval)) % 24).view(1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])

        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2)
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2)
        
        sin_seas_emb = torch.sin(torch.pi * t_emb/ (12 * 365) - torch.pi / 2)
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2)

        day_emb = torch.cat([sin_t_emb, cos_t_emb], 1)
        seas_emb = torch.cat([sin_seas_emb, cos_seas_emb], 1)

        t_cyc_emb = torch.cat([day_emb, seas_emb], 1)
        
        t_cyc_emb_expanded = t_cyc_emb.unsqueeze(2)
        pos_feats_expanded = pos_feats.unsqueeze(1)
        pos_time_ft = (t_cyc_emb_expanded * pos_feats_expanded).view(t_cyc_emb.shape[0], -1, t_cyc_emb.shape[2], t_cyc_emb.shape[3])

        comb_rep = torch.cat([t_emb / 24, day_emb, seas_emb, nabla_x, v, x, new_lat_lon, pos_feats, pos_time_ft], 1)

        dv = self.model(comb_rep)

        dv += self.linear_model(comb_rep.reshape(comb_rep.shape[0], comb_rep.shape[2], comb_rep.shape[3], -1)).reshape(*dv.shape)

        adv1 = vx * x_grad_x + vy * x_grad_y
        adv2 = x * (torch.gradient(vx, dim=3)[0] + torch.gradient(vy, dim=2)[0])

        x = adv1 + adv2

        return torch.cat([x, dv, new_lat_lon, pos_feats], 1)


    def forward(self, x, y, predict_range, variables, out_variables, metric, lat, lon, vis_noise=False, epoch=0):

        v_net_input = torch.cat([x, torch.gradient(x, dim=3)[0], torch.gradient(x, dim=2)[0]], 1)

        v_output = self.v_net(v_net_input)
        vx, vy = v_output[:, :x.shape[1]], v_output[:, x.shape[1]:]

        new_lat = torch.tensor(lat).float().expand(x.shape[3], x.shape[2]).T.to(x.device).expand(x.shape[0], 1, x.shape[2], x.shape[3]) * torch.pi / 180
        new_lon = torch.tensor(lon).float().expand(x.shape[2], x.shape[3]).to(x.device).expand(x.shape[0], 1, x.shape[2], x.shape[3]) * torch.pi / 180

        new_lat_lon = torch.cat([new_lat, new_lon], 1)

        cos_lat_map, sin_lat_map = torch.cos(new_lat), torch.sin(new_lat)
        cos_lon_map, sin_lon_map = torch.cos(new_lon), torch.sin(new_lon)

        pos_feats = torch.cat([cos_lat_map, cos_lon_map, sin_lat_map, sin_lon_map, sin_lat_map * cos_lon_map, sin_lat_map * sin_lon_map], 1)

        ode_x = torch.cat([x, vx, vy, new_lat_lon, pos_feats], 1)

        new_time_steps = torch.linspace(int(predict_range[0] / self.time_steps), int(predict_range[0]), self.time_steps).float().to(x.device) * self.time_interval

        final_result = odeint(self.pde, ode_x, new_time_steps, method=self.method, rtol=self.rtol, atol=self.atol)

        preds = final_result[:, :, :len(self.default_vars)]

        out_ids = self.get_var_ids(tuple(out_variables), preds.device)
        y_ = y.permute(1,0,2,3,4)[torch.linspace(int(predict_range[0] / self.time_steps), int(predict_range[0]), self.time_steps).long() - 1]

        if self._check_for_nan(preds, "ode"):
            if metric is None:
                loss = None
                if vis_noise:
                    return preds[:, :, out_ids], preds[:, :, out_ids], preds[:, :, out_ids]
            else:
                preds = preds[:, :, out_ids]
                loss = [m(preds, preds, preds, y_, out_variables, lat) for m in metric]
                return loss, preds

        if self.use_err:
            noise_x = torch.cat([preds, new_lat_lon.expand(preds.shape[0], *new_lat_lon.shape), pos_feats.expand(preds.shape[0], *pos_feats.shape)], 2)
            if self.err_with_x:
                noise_x = torch.cat([noise_x, x.expand(preds.shape[0], *x.shape)], 2)
            if self.err_with_v:
                noise_x = torch.cat([noise_x, v_output.expand(preds.shape[0], *v_output.shape)], 2)

            if self.err_type == "2D":
                noise_x = noise_x.reshape(-1, *noise_x.shape[2:])
                noise_output = self.noise_model(noise_x)
                noise_output = noise_output.view(preds.shape[0], -1, *noise_output.shape[1:])
            elif self.err_type == "2DTime":
                noise_x = noise_x.reshape(-1, *noise_x.shape[2:])

                time_embedding = torch.repeat_interleave(torch.linspace(int(predict_range[0] / self.time_steps), int(predict_range[0]), self.time_steps, device=preds.device), preds.shape[1])

                noise_output = self.noise_model(noise_x, time_embedding)

                noise_output = noise_output.view(preds.shape[0], -1, *noise_output.shape[1:])
            elif self.err_type == "DiT":
                noise_x = noise_x.reshape(-1, *noise_x.shape[2:])

                time_embedding = torch.repeat_interleave(torch.linspace(int(predict_range[0] / self.time_steps), int(predict_range[0]), self.time_steps, device=preds.device), preds.shape[1])

                noise_output = self.noise_model(noise_x, time_embedding)
                noise_output = noise_output.view(preds.shape[0], -1, *noise_output.shape[1:])
            elif self.err_type == "3D":
                noise_output = self.noise_model(noise_x)
            elif self.err_type == "2+1D":
                noise_output = self.noise_model(noise_x)
            elif self.err_type == "vit":
                noise_x = noise_x.reshape(-1, *noise_x.shape[2:])
                noise_output = self.noise_model(noise_x)
                noise_output = noise_output.view(preds.shape[0], -1, *noise_output.shape[1:])

            if torch.isnan(noise_output).any():
                print("noise nan \n")

            if self._check_for_nan(noise_output, "noise net"):
                if metric is None:
                    loss = None
                    if vis_noise:
                        return preds[:, :, out_ids], preds[:, :, out_ids], preds[:, :, out_ids]
                else:
                    loss = [m(preds[:, :, out_ids], preds[:, :, out_ids], preds[:, :, out_ids], y_, out_variables, lat) for m in metric]
                    return loss, preds

            final_preds = preds + noise_output[:, :, :len(self.default_vars)]
        else:
            final_preds = preds.clone()
            noise_output = preds.clone()

        final_preds = final_preds[:, :, out_ids]

        if metric is None:
            # preds = preds[-1]
            loss = None
            if vis_noise:
                return final_preds, preds[:, :, out_ids], noise_output[:, :, :len(self.default_vars)][:, :, out_ids]
        else:
            loss = [m(final_preds, preds[:, :, out_ids], noise_output[:, :, :len(self.default_vars)][:, :, out_ids], y_, out_variables, lat, gradient_loss=self.gradient_loss, epoch=epoch) for m in metric]

        return loss, final_preds

    def evaluate(self, x, y, predict_range, variables, out_variables, transform, metrics, lat, lon, clim, log_postfix):
        preds, ode_preds, noise_preds = self.forward(x, y, predict_range, variables, out_variables, metric=None, lat=lat, lon=lon, vis_noise=True)

        ratio = int(predict_range.mean()) // preds.shape[0]

        loss_dict = []

        for pred_range in self.predict_list:
            if pred_range < 24:
                log_postfix = f"{pred_range}_hours"
            else:
                days = pred_range // 24
                if pred_range > days * 24:
                    log_postfix = f"{days}_days_{pred_range - days * 24}_hours"
                else:
                    log_postfix = f"{days}_days"

            steps = pred_range // ratio
            
            dic_list = [m(preds[steps - 1], y.permute(1,0,2,3,4)[pred_range - 1], transform, out_variables, lat, clim, log_postfix) for m in metrics]

            if pred_range != int(predict_range.mean()):
                for dic in dic_list:
                    dic.pop('w_rmse', None)

            loss_dict += dic_list

        return loss_dict, preds[-1], ode_preds[-1], noise_preds[-1]

    def _check_for_nan(self, tensor: torch.Tensor, step: str) -> bool:
        has_nan = torch.isnan(tensor).any().float()
        
        if dist.is_initialized():
            dist.all_reduce(has_nan, op=dist.ReduceOp.SUM)
        
        if has_nan > 0:
            if dist.is_initialized():
                rank = dist.get_rank()
                print(f"NaN detected on GPU {rank} at step: {step}")
                dist.barrier()
            else:
                print(f"NaN detected in single-GPU training at step: {step}")
            return True
        return False
