import math
import random

import torch
import numpy
from scipy.spatial import distance_matrix


def get_random_problems(batch_size, problem_size):
    depot_x = 5.622153766066174
    depot_y = 52.0308709742657
    depot_xy = torch.tensor([depot_x,depot_y]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)
    depot_time_window = torch.tensor([0, 1]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)

    node_x = 4+torch.rand(size=(batch_size, problem_size,1))*3
    node_y =  51+torch.rand(size=(batch_size, problem_size,1))*2.5
    node_xy = torch.cat((node_x, node_y), dim=2)
    # shape: (batch, problem, 2)

    demand_scaler = 25000

    rate = 1/382
    node_demand = torch.distributions.Exponential(rate).sample((batch_size, problem_size)) / demand_scaler
    # shape: (batch, problem)

    p = 0.25
    # Create tensor of probabilities
    samples = torch.rand((batch_size, problem_size, 1)) < p

    tw_scalar = 14
    horizon_start = 5
    lower_tw = torch.tensor([5],dtype=torch.float32).repeat(batch_size, problem_size, 1)
    upper_tw = torch.tensor([19],dtype=torch.float32).repeat(batch_size, problem_size, 1)
    lower_tw[samples] = (torch.randint(15, 21, (batch_size, problem_size, 1))/ 2)[samples]
    upper_tw[samples] = (torch.randint(30, 38, (batch_size, problem_size, 1))/ 2)[samples]
    time_windows = torch.cat((lower_tw, upper_tw), dim=2)
    time_windows = (time_windows - horizon_start) / tw_scalar
    print(time_windows[0,0,:10])

    empty_tensor = torch.empty(batch_size, problem_size)
    service_times = torch.nn.init.trunc_normal_(empty_tensor, mean=0.23, std=0.24,
                                                a=0.05, b=0.6)

    p=0.02
    # Create tensor of probabilities
    samples = torch.rand((batch_size, problem_size)) < p
    service_times[samples] = (torch.randint(100,151, (batch_size,problem_size))/100)[samples]
    service_times = service_times/tw_scalar

    travel_times = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    prices = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    duals = torch.zeros((batch_size, problem_size+1))
    for x in range(batch_size):
        coords = torch.cat((depot_xy[x], node_xy[x]), 0)
        travel_times[x] = torch.cdist(coords, coords, p=2)
        travel_times[x].fill_diagonal_(0)
        # travel_times[x, travel_times[x] < 0.5] = travel_times[x, travel_times[x] < 0.5]*5
        duals[x] = create_duals(travel_times[x])
        prices[x] = (travel_times[x] -duals[x]) * -1
        prices[x].fill_diagonal_(0)
        min_val = torch.min(prices[x]).item()
        max_val = torch.max(prices[x]).item()
        prices[x] = prices[x] / max(abs(max_val), abs(min_val))

    travel_times = travel_times / tw_scalar
    duals = duals[:,1:] / tw_scalar
    time_windows = repair_time_windows(travel_times, time_windows, service_times,
                                       tw_scalar, 4)

    depot_xy[:,:,0] = (depot_xy[:,:,0]-4)/3
    depot_xy[:, :, 1] = (depot_xy[:, :, 1] - 51) / 2.5
    node_xy[:, :, 0] = (node_xy[:, :, 0] - 4) / 3
    node_xy[:, :, 1] = (node_xy[:, :, 1] - 51) / 2.5


    return depot_xy, node_xy, node_demand, time_windows, depot_time_window, duals, service_times, travel_times, prices


def create_duals(time_matrix):
    problem_size = time_matrix.shape[1]-1
    duals = torch.zeros(size=(problem_size+1,), dtype=torch.float32)
    indices = list(range(1, problem_size+1))
    scaler = 0.2 + 0.9 * torch.rand([]) #0.75*torch.rand([])
    non_zeros = random.randint(problem_size / 2, problem_size)
    chosen = random.sample(indices, non_zeros)
    max_travel_times, _ = torch.max(time_matrix,dim=0)
    randoms = torch.rand(size=(non_zeros,))
    duals[chosen] = max_travel_times[chosen] * scaler * randoms
    return duals

def repair_time_windows(travel_times, time_windows, service_times, tw_scalar,
                        min_tw_width, factor=0.0001):

    scaled_tw_width = min_tw_width / tw_scalar
    problem_size = travel_times.shape[1] - 1
    batch_size = travel_times.shape[0]

    latest_possible_arrivals = torch.ones(batch_size, problem_size) - travel_times[:, 1:, 0] - service_times
    latest_possible_arrivals -= factor
    time_windows[:, :, 1] = torch.minimum(time_windows[:, :, 1], latest_possible_arrivals)
    mask = time_windows[:, :, 1] - time_windows[:, :, 0] < scaled_tw_width

    time_windows[:, :, 0][mask] = time_windows[:, :, 1][mask] - scaled_tw_width
    return time_windows

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
