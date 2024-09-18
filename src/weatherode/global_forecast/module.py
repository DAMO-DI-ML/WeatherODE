# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import wandb
import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from weatherode.ode import WeatherODE
from weatherode.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from weatherode.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    lat_weighted_mse_velocity_guess
)
from weatherode.utils.pos_embed import interpolate_pos_embed

from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the WeatherODE model.

    Args:
        net (WeatherODE): WeatherODE model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: WeatherODE,
        pretrained_path: str = "",
        lr: float = 5e-4,
        ode_lr: float = 5e-5,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        gradient_clip_val: float = 0.5,
        gradient_clip_algorithm: str = "value",
        train_noise_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.skip_optimization = False
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        # interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        # if self.net.parallel_patch_embed:
        #     if "token_embeds.proj_weights" not in checkpoint_model.keys():
        #         raise ValueError(
        #             "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
        #         )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        x, y, predict_ranges, variables, out_variables = batch

        # init_vx, init_vy = optimize_vel(x, prev_x, self.kernel)

        loss_dict, _ = self.net.forward(x, y, predict_ranges, variables, out_variables, [lat_weighted_mse_velocity_guess], lat=self.lat, lon=self.lon, epoch=self.current_epoch)
        loss_dict = loss_dict[0]

        # check nan
        has_nan = False
        for var, loss_value in loss_dict.items():
            if torch.isnan(loss_value).any():
                has_nan = True
                break

        # sum nan
        has_nan_tensor = torch.tensor(float(has_nan), device=self.device)
        torch.distributed.all_reduce(has_nan_tensor, op=torch.distributed.ReduceOp.SUM)
        
        if has_nan_tensor.item() > 0:
            self.log("train/has_nan", True, on_step=True, on_epoch=False, prog_bar=True)
            self.skip_optimization = True
            return loss_dict["loss"] #torch.tensor(1., device=self.device, requires_grad=True).to(torch.float32)
        
        self.log("train/has_nan", False, on_step=True, on_epoch=False, prog_bar=True)
        self.skip_optimization = False

        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]

        return loss

    def plot_weather_maps(self, preds, out_variables, batch_idx, lat, lon, type="gt", test=False):
        """
        Plots weather maps for given predictions and logs them to the experiment logger.

        Args:
            preds (torch.Tensor): The predictions tensor of shape [batch_size, 5, 32, 64].
            out_variables (list): List of variable names for the channels.
            batch_idx (int): The batch index.
            logger: The experiment logger.
            lat (np.array): Latitude values.
            lon (np.array): Longitude values.
        """
        prefix = "test_images" if test else "val_images"

        batch_size, num_vars, height, width = preds.shape
        for var_idx in range(num_vars):
            fig, ax = plt.subplots()
            m = Basemap(projection='cyl', resolution='c', ax=ax,
                        llcrnrlat=lat.min(), urcrnrlat=lat.max(),
                        llcrnrlon=lon.min(), urcrnrlon=lon.max())
            m.drawcoastlines()
            m.drawcountries()
            m.drawmapboundary()
            m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])
            m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])

            data = preds[0, var_idx].cpu().detach().numpy()
            
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            xi, yi = m(lon_grid, lat_grid)

            # Interpolate the data if needed (you can adjust the method and resolution)
            data = np.interp(data, (data.min(), data.max()), (0, 1))

            # Plot the data
            cs = m.pcolormesh(xi, yi, data, cmap='RdBu')
            fig.colorbar(cs, ax=ax, orientation='vertical', label=out_variables[var_idx])

            ax.set_title(f"{type}_{out_variables[var_idx]}")

            # Log the figure to the experiment logger
            try:
                self.logger.experiment.log({f"{prefix}/{out_variables[var_idx]}_{type}_{batch_idx}": wandb.Image(fig)}, step=self.global_step)
            except:
                self.logger.experiment.add_figure(f"{prefix}/{out_variables[var_idx]}_{type}_{batch_idx}", fig, global_step=self.global_step)
            plt.close(fig)

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, predict_ranges, variables, out_variables = batch

        # init_vx, init_vy = optimize_vel(x, prev_x, self.kernel)

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts, preds, ode_preds, noise_preds = self.net.evaluate(
            x,
            y,
            predict_ranges,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            lon=self.lon,
            clim=self.val_clim,
            log_postfix=log_postfix,
        )

        if batch_idx % 50 == 0:
            self.plot_weather_maps(y[:, -1], out_variables, batch_idx, self.lat, self.lon, type="gt")
            self.plot_weather_maps(ode_preds, out_variables, batch_idx, self.lat, self.lon, type="ode")
            self.plot_weather_maps(noise_preds, out_variables, batch_idx, self.lat, self.lon, type="noise")

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, predict_ranges, variables, out_variables = batch

        # init_vx, init_vy = optimize_vel(x, prev_x, self.kernel)

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts, preds, ode_preds, noise_preds = self.net.evaluate(
            x,
            y,
            predict_ranges,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            lon=self.lon,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )

        if batch_idx % 50 == 0:
            self.plot_weather_maps(y[:, -1], out_variables, batch_idx, self.lat, self.lon, type="gt", test=True)
            self.plot_weather_maps(ode_preds, out_variables, batch_idx, self.lat, self.lon, type="ode", test=True)
            self.plot_weather_maps(noise_preds, out_variables, batch_idx, self.lat, self.lon, type="noise", test=True)

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    def configure_optimizers(self):
        decay = []
        no_decay = []

        ode = []
        noise_net = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                if "net.model" in name:
                    ode.append(m)
                else:
                    decay.append(m)
            
            if self.hparams.train_noise_only:
                if "net.noise_model" in name:
                    noise_net.append(m)
                else:
                    m.requires_grad = False

        if self.hparams.train_noise_only:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": noise_net,
                        "lr": self.hparams.lr,
                        "betas": (self.hparams.beta_1, self.hparams.beta_2),
                        "weight_decay": self.hparams.weight_decay,
                    }
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": ode,
                        "lr": self.hparams.ode_lr,
                        "betas": (self.hparams.beta_1, self.hparams.beta_2),
                        "weight_decay": self.hparams.weight_decay,
                    },
                    {
                        "params": decay,
                        "lr": self.hparams.lr,
                        "betas": (self.hparams.beta_1, self.hparams.beta_2),
                        "weight_decay": self.hparams.weight_decay,
                    },
                    {
                        "params": no_decay,
                        "lr": self.hparams.lr,
                        "betas": (self.hparams.beta_1, self.hparams.beta_2),
                        "weight_decay": 0,
                    },
                ]
            )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # check loss

        if self.skip_optimization:
            optimizer.zero_grad()
            optimizer_closure()
            return

        optimizer.step(closure=optimizer_closure)