import torch
from weatherode.ode import WeatherODE
from torchdiffeq import odeint
import torch.distributed as dist

class RegionalWeatherODE(WeatherODE):
    def __init__(
        self,
        default_vars,
        method,
        img_size=[32, 64],
        patch_size=2,
        layers=[5, 5, 3, 2],
        hidden=[512, 128, 64],
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
        super().__init__(default_vars, method, img_size, patch_size, layers, hidden, depth, use_err, err_type, err_with_x, err_with_v, err_with_std, drop_rate, time_steps, time_interval, rtol, atol, predict_list, gradient_loss)

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
                    return x.expand(self.time_steps, *x.shape)[:, :, out_ids], x.expand(self.time_steps, *x.shape)[:, :, out_ids], x.expand(self.time_steps, *x.shape)[:, :, out_ids]
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

        final_preds = final_preds[:, :, out_ids]

        if metric is None:
            # preds = preds[-1]
            loss = None
            if vis_noise:
                return final_preds, preds[:, :, out_ids], noise_output[:, :, :len(self.default_vars)][:, :, out_ids]
        else:
            loss = [m(final_preds, preds[:, :, out_ids], noise_output[:, :, :len(self.default_vars)][:, :, out_ids], y_, out_variables, lat, gradient_loss=self.gradient_loss, epoch=epoch) for m in metric]

        return loss, final_preds

    def evaluate(self, x, y, predict_range, variables, out_variables, transform, metrics, lat, lon, clim, log_postfix, region_info):
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        x = x[:, :, min_h:max_h+1, min_w:max_w+1]
        y = y[:, :, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]
        lon = lon[min_w:max_w+1]
        clim = clim[:, min_h:max_h+1, min_w:max_w+1]

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
