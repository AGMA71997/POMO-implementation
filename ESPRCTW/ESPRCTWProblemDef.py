import math
import random

import torch
import numpy
from scipy.spatial import distance_matrix


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 300:
        demand_scaler = 100
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / demand_scaler
    # shape: (batch, problem)
    depot_time_window = torch.tensor([0, 1]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)
    tw_scalar = 18
    lower_tw = torch.randint(0, 17, (batch_size, problem_size))
    tw_width = torch.randint(2, 9, (batch_size, problem_size))
    upper_tw = tw_width + lower_tw
    upper_tw = torch.minimum(upper_tw, torch.ones(upper_tw.shape) * tw_scalar)
    time_windows = torch.zeros((batch_size, problem_size, 2))
    time_windows[:, :, 0] = lower_tw
    time_windows[:, :, 1] = upper_tw
    time_windows = time_windows / tw_scalar
    service_times = (torch.rand((batch_size, problem_size)) * 0.3 + 0.2) / tw_scalar
    travel_times = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    prices = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    duals = torch.zeros((batch_size, problem_size))
    for x in range(batch_size):
        coords = torch.cat((depot_xy[x], node_xy[x]), 0)
        travel_times[x] = torch.FloatTensor(distance_matrix(coords, coords))
        travel_times[x].fill_diagonal_(0)
        duals[x] = create_duals(1, problem_size, travel_times[x:x + 1])[0]
        prices[x] = (travel_times[x] - torch.cat((torch.tensor([0]), duals[x]), 0)) * -1
        prices[x].fill_diagonal_(0)
        min_val = torch.min(prices[x])
        max_val = torch.max(prices[x])
        prices[x] = prices[x] / max(abs(max_val), abs(min_val))

    travel_times = travel_times / tw_scalar
    duals = duals / tw_scalar

    return depot_xy, node_xy, node_demand, time_windows, depot_time_window, duals, service_times, travel_times, prices


def create_duals(batch_size, problem_size, time_matrix):
    duals = torch.zeros(size=(batch_size, problem_size), dtype=torch.float32)
    for x in range(batch_size):
        scaler = 0.2 + 0.9 * numpy.random.random()
        non_zeros = numpy.random.randint(problem_size / 2, problem_size + 1)
        indices = list(range(problem_size))
        chosen = random.sample(indices, non_zeros)
        for index in chosen:
            duals[x, index] = torch.max(time_matrix[x, :, index + 1]) * scaler * numpy.random.random()

    return duals


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems
