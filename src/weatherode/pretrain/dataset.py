# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                num_nodes = 1
                # num_nodes = int(os.environ.get("NODES", None))
                num_gpus_per_node = int(world_size / num_nodes)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, max_predict_range: int = 6, random_lead_time: bool = False, hrs_each_step: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        
        self.max_predict_range = max_predict_range
        self.random_lead_time = random_lead_time
        self.hrs_each_step = hrs_each_step

    def __iter__(self):
        for data, variables, out_variables in self.dataset:
            # [1095, 5, 32, 64]
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)

            # [1095, 5, 32, 64]
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x[: -self.max_predict_range]  # N, C, H, W

            # t-2, t-1 and t
            # prev_inputs = torch.stack([inputs[i:i+3] for i in range(inputs.size(0) - 2)], dim=0)

            # 2 for torchcubicspline
            # inputs = inputs[2:]

            if self.random_lead_time:
                predict_ranges = torch.randint(low=1, high=self.max_predict_range, size=(inputs.shape[0],))
            else:
                predict_ranges = torch.ones(inputs.shape[0]).to(torch.long) * self.max_predict_range
            # lead_times = self.hrs_each_step * predict_ranges / 100
            # lead_times = lead_times.to(inputs.dtype)

            base_index = torch.arange(inputs.shape[0]).unsqueeze(0)

            range_index = torch.arange(1, self.max_predict_range + 1).unsqueeze(1)

            # 2 for torchcubicspline
            final_index = base_index + range_index

            outputs = y[final_index].permute(1, 0, 2, 3, 4)

            yield inputs, outputs, predict_ranges.to(inputs.dtype), variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.region_info = region_info

        # self.lat = lat 
        # self.lon = lon
        # self.set_gauss_kernel()
        # breakpoint()
        # for (inp, prev_inp, out, predict_ranges, variables, out_variables) in self.dataset:
        #     optimize_vel(inp, prev_inp, self.kernel)

    def __iter__(self):
        for (inp, out, predict_ranges, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.region_info is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), predict_ranges[i], variables, out_variables, self.region_info
                else:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), predict_ranges[i], variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
