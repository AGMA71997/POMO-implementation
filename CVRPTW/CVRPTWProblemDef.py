import torch
import numpy as np


# def get_random_problems(batch_size, problem_size):
#
#     depot_xy = torch.rand(size=(batch_size, 1, 2))
#     # shape: (batch, 1, 2)
#
#     node_xy = torch.rand(size=(batch_size, problem_size, 2))
#     # shape: (batch, problem, 2)
#
#     if problem_size == 20:
#         demand_scaler = 30
#     elif problem_size == 50:
#         demand_scaler = 40
#     elif problem_size == 100:
#         demand_scaler = 50
#     else:
#         raise NotImplementedError
#
#     node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
#     # shape: (batch, problem)
#
#     node_tw_start = torch.randint(0, 21, size=(batch_size, problem_size, 1))
#     node_tw_end = node_tw_start + torch.randint(4, 9, size=(batch_size, problem_size, 1))
#     node_tw = torch.cat((node_tw_start, node_tw_end), dim=2)
#     node_tw = node_tw
#
#     service_t = 1
#
#     return depot_xy, node_xy, node_demand, node_tw, service_t
#
#
# def get_random_problems_2(batch_size, problem_size):
#     service_window = 1000
#     service_duration = 10
#     time_factor = 100
#     tw_expansion = 3
#
#     # Sample locations
#     depot_xy = torch.rand(size=(batch_size, 1, 2))
#     node_xy = torch.rand(size=(batch_size, problem_size, 2))
#
#     # Distance from the nodes to the depot
#     traveling_time = torch.linalg.vector_norm(depot_xy - node_xy, dim=-1) * time_factor
#
#     # TW start needs to be feasibly reachable directly from depot
#     tw_start_min = torch.ceil(traveling_time) + 1
#
#     # TW end needs to be early enough to perform service and return to depot until end of service window
#     tw_end_max = service_window - torch.ceil(traveling_time + service_duration) - 1
#
#     # Horizon allows for the feasibility of reaching nodes/returning from nodes within the global TW (service window)
#     horizon = torch.stack([tw_start_min, tw_end_max], dim=-1)
#
#     # Sample time windows start
#     tw_start = tw_start_min + torch.round((horizon[:, :, 1] - horizon[:, :, 0]) * torch.rand(batch_size, problem_size))
#
#     # Sample time windows end
#     epsilon = torch.clamp(torch.abs(torch.randn(batch_size, problem_size)), min=1 / time_factor)
#     tw_end = torch.clamp(tw_start + tw_expansion * time_factor * epsilon, max=tw_end_max)
#
#     node_tw = torch.stack([tw_start, tw_end], dim=-1)
#
#     # Sample node demands
#     node_demand = torch.clamp(torch.normal(mean=15, std=10, size=[batch_size, problem_size]).abs().round(), min=1, max=42)
#
#     # Scale node_tw and the service duration based on the time factor
#     node_tw /= time_factor
#     service_duration /= time_factor
#
#     # Scale node demand based on vehicle capacity
#     if problem_size == 20:
#         demand_scaler = 500
#     elif problem_size == 50:
#         demand_scaler = 750
#     elif problem_size == 100:
#         demand_scaler = 1000
#     else:
#         raise NotImplementedError
#
#     node_demand /= demand_scaler
#
#     return depot_xy, node_xy, node_demand, node_tw, service_duration
#
# def get_random_problems_3(batch_size, problem_size):
#     service_window = 10
#     service_duration = 0.1
#     tw_expansion = 3
#     epsilon = 0.01
#
#     # Sample locations
#     depot_xy = torch.rand(size=(batch_size, 2))
#     node_xy = torch.rand(size=(batch_size, problem_size, 2))
#
#     # Distance from the nodes to the depot
#     traveling_time = torch.linalg.vector_norm(depot_xy[:, None] - node_xy, dim=-1)
#
#     # TW start needs to be feasibly reachable directly from depot
#     tw_start_min = traveling_time + epsilon
#
#     # TW end needs to be early enough to perform service and return to depot until end of service window
#     tw_end_max = service_window - traveling_time + service_duration - epsilon
#
#     # Horizon allows for the feasibility of reaching nodes/returning from nodes within the global TW (service window)
#     horizon = torch.stack([tw_start_min, tw_end_max], dim=-1)
#
#     # Sample time windows start
#     tw_start = tw_start_min + (horizon[:, :, 1] - horizon[:, :, 0]) * torch.rand(batch_size, problem_size)
#
#     # Sample time windows end
#     epsilon = torch.clamp(torch.abs(torch.randn(batch_size, problem_size)), min=1 / time_factor)
#     tw_end = torch.clamp(tw_start + tw_expansion * time_factor * epsilon, max=tw_end_max)
#
#     node_tw = torch.stack([tw_start, tw_end], dim=-1)
#
#     # Sample node demands
#     node_demand = torch.clamp(torch.normal(mean=15, std=10, size=[batch_size, problem_size]).abs().round(), min=1, max=42)
#
#     # Scale node_tw and the service duration based on the time factor
#     node_tw /= time_factor
#     service_duration /= time_factor
#
#     # Scale node demand based on vehicle capacity
#     if problem_size == 20:
#         demand_scaler = 500
#     elif problem_size == 50:
#         demand_scaler = 750
#     elif problem_size == 100:
#         demand_scaler = 1000
#     else:
#         raise NotImplementedError
#
#     node_demand /= demand_scaler
#
#     return depot_xy, node_xy, node_demand, node_tw, service_duration
#
# def get_random_problems_r101(batch_size, problem_size, scale_values=True):
#     service_window = 230
#     service_duration = 10
#     time_window_size = 10
#
#     # Scale node demand based on vehicle capacity
#     if problem_size == 20:
#         capacity = 70
#     elif problem_size == 50:
#         capacity = 100
#     elif problem_size == 100:
#         capacity = 200
#     else:
#         raise NotImplementedError
#
#
#     # Sample locations
#     depot_xy = 45 + torch.randint(0, 11, size=(batch_size, 1, 2))  # The depot is always in the center ([0.45, 0.55]). Otherwise not all customers can be reached with a service window of 230
#     node_xy = torch.randint(0, 100, size=(batch_size, problem_size, 2))
#
#     # Distance from the nodes to the depot
#     traveling_time = torch.linalg.vector_norm((depot_xy - node_xy).float(), dim=-1)
#
#     # TW start needs to be feasibly reachable directly from depot
#     tw_start_min = torch.ceil(traveling_time) + 1
#
#     # TW end needs to be early enough to perform service and return to depot until end of service window
#     tw_end_max = service_window - torch.ceil(traveling_time + service_duration) - 1
#
#     # Horizon allows for the feasibility of reaching nodes/returning from nodes within the global TW (service window)
#     #horizon = torch.stack([tw_start_min, tw_end_max], dim=-1)
#
#     # Sample time windows center
#     tw_center = tw_start_min + torch.round((tw_end_max - tw_start_min) * torch.rand(batch_size, problem_size))
#
#     # Define time window start and end
#     tw_start = tw_center - time_window_size // 2
#     tw_end = tw_center + time_window_size // 2
#     tw_end = torch.clamp(tw_end, max=tw_end_max)  # Set tw_end so that the vehicle always returns to the depot before
#                                                   # the end of the service_window
#
#     if (tw_end < tw_start_min).any():
#         print("h")
#
#     node_tw = torch.stack([tw_start, tw_end], dim=-1)
#
#     depot_tw = torch.Tensor([[0, service_window]]).repeat(batch_size, 1)
#
#     # Sample node demands
#     dis = torch.distributions.beta.Beta(1.4, 3)
#     node_demand = torch.round(dis.rsample([batch_size, problem_size]) * 45 + 1)
#
#     if scale_values:
#         node_demand /= capacity
#
#     return depot_xy, node_xy, node_demand, depot_tw, node_tw, service_duration, capacity
#
def get_random_problems_dummy(batch_size, problem_size):
    service_window = 24000
    service_duration = 1

    # Sample locations
    depot_xy = torch.randint(0, 1000, size=(batch_size, 1, 2))
    node_xy = torch.randint(0, 1000, size=(batch_size, problem_size, 2))

    # Sample time windows start
    tw_start = torch.zeros(batch_size, problem_size)

    tw_end = torch.ones(batch_size, problem_size) * 24000

    node_tw = torch.stack([tw_start, tw_end], dim=-1)
    depot_tw = torch.IntTensor([[0, service_window]]).repeat(batch_size, 1)

    # Scale node demand based on vehicle capacity
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size))
    capacity = torch.ones(batch_size) * demand_scaler

    return depot_xy, node_xy, node_demand, depot_tw, node_tw, service_duration, capacity


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
