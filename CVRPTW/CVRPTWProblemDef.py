import torch
import numpy as np
from scipy.spatial import distance_matrix

tw_scalar = 18
min_tw_width = 2
max_tw_width = 9


def get_random_problems(batch_size, problem_size):
    depot_x = 5.622153766066174
    depot_y = 52.0308709742657
    depot_xy = torch.tensor([depot_x / 7, depot_y / 53.5]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)

    node_x = 4 + torch.rand(size=(batch_size, problem_size, 1)) * 3
    node_y = 51 + torch.rand(size=(batch_size, problem_size, 1)) * 2.5
    node_xy = torch.cat((node_x / 7, node_y / 53.5), dim=2)
    # shape: (batch, problem, 2)

    demand_scaler = 25000

    rate = 1 / 382
    node_demand = torch.distributions.Exponential(rate).sample((batch_size, problem_size)) / demand_scaler
    # shape: (batch, problem)

    p = 0.25
    # Create tensor of probabilities
    probs = torch.full((batch_size, problem_size, 1), p)  # 3x4 tensor with all values = p
    # Sample Bernoulli random variables (0 or 1)
    samples = torch.bernoulli(probs).long()

    tw_scalar = 14
    horizon_start = 5
    lower_tw = torch.tensor([5], dtype=torch.float32).repeat(batch_size, problem_size, 1)
    upper_tw = torch.tensor([19], dtype=torch.float32).repeat(batch_size, problem_size, 1)
    lower_tw[samples] = (torch.randint(15, 20, (batch_size, problem_size, 1)) / 2)[samples]
    upper_tw[samples] = (torch.randint(30, 37, (batch_size, problem_size, 1)) / 2)[samples]
    time_windows = torch.cat((lower_tw, upper_tw), dim=2)
    time_windows = (time_windows - horizon_start) / tw_scalar

    empty_tensor = torch.empty(batch_size, problem_size)
    service_times = torch.nn.init.trunc_normal_(empty_tensor, mean=0.23, std=0.24,
                                                a=0, b=1.5) / tw_scalar

    travel_times = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    for x in range(batch_size):
        coords = torch.cat((depot_xy[x], node_xy[x]), 0)
        travel_times[x] = torch.cdist(coords, coords, p=1)
        travel_times[x].fill_diagonal_(0)
        travel_times[x, travel_times[x] < 0.5] = travel_times[x, travel_times[x] < 0.5] * 5

    travel_times = travel_times / tw_scalar

    return depot_xy, node_xy, node_demand, time_windows, service_times, travel_times


def repair_time_windows(travel_times, time_windows, service_times,
                        scaled_tw_width=min_tw_width / tw_scalar, factor=0.0001):
    problem_size = travel_times.shape[1] - 1
    batch_size = travel_times.shape[0]

    latest_possible_arrivals = torch.ones(batch_size, problem_size) - travel_times[:, 1:, 0] - service_times
    latest_possible_arrivals -= factor
    time_windows[:, :, 1] = torch.minimum(time_windows[:, :, 1], latest_possible_arrivals)
    mask = time_windows[:, :, 1] - time_windows[:, :, 0] < scaled_tw_width

    time_windows[:, :, 0][mask] = time_windows[:, :, 1][mask] - scaled_tw_width
    return time_windows


def get_random_problems_from_data(depot_xy, node_xy, node_demand, augment=True):
    service_window = 2400  # v1: 2300 v2: 2400
    service_duration = 50  # v1: 100 v2: 50
    time_window_size = 500

    batch_size = node_xy.shape[0]
    problem_size = node_xy.shape[1]

    # Distance from the nodes to the depot
    traveling_time = torch.linalg.vector_norm((depot_xy - node_xy).float(), dim=-1)

    # TW start needs to be feasibly reachable directly from depot
    tw_start_min = torch.ceil(traveling_time) + 1

    # TW end needs to be early enough to perform service and return to depot until end of service window
    tw_end_max = service_window - torch.ceil(traveling_time + service_duration) - 1

    # Sample time windows center
    tw_center = tw_start_min + torch.round((tw_end_max - tw_start_min) * torch.rand(batch_size, problem_size))

    # Define time window start and end
    tw_start = tw_center - time_window_size // 2
    tw_end = tw_center + time_window_size // 2

    tw_start = torch.clamp(tw_start, min=tw_start_min)
    tw_end = torch.clamp(tw_end, max=tw_end_max)

    node_tw = torch.stack([tw_start, tw_end], dim=-1).int()
    depot_tw = torch.IntTensor([[0, service_window]]).repeat(batch_size, 1)

    return depot_xy, node_xy, node_demand, depot_tw, node_tw, service_duration


def augment_xy_data_by_8_fold(xy_data, grid_size):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((grid_size - x, y), dim=2)
    dat3 = torch.cat((x, grid_size - y), dim=2)
    dat4 = torch.cat((grid_size - x, grid_size - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((grid_size - y, x), dim=2)
    dat7 = torch.cat((y, grid_size - x), dim=2)
    dat8 = torch.cat((grid_size - y, grid_size - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data
