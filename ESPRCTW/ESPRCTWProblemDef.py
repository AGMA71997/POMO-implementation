import torch
import numpy


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
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)
    depot_time_window = torch.tensor([0, 1]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)
    tw_scalar = 18
    time_windows = torch.tensor(create_time_windows(batch_size, problem_size, tw_scalar)) / float(tw_scalar)
    service_times = create_service_times(batch_size, problem_size) / float(tw_scalar)
    travel_times = create_time_matrix(batch_size, problem_size, node_xy, depot_xy) / float(tw_scalar)
    duals = create_duals(batch_size, problem_size)

    travel_times = torch.tensor(travel_times)
    duals = torch.tensor(duals, dtype=torch.float32) / float(duals.max())
    service_times = torch.tensor(service_times, dtype=torch.float32)

    return depot_xy, node_xy, node_demand, time_windows, depot_time_window, duals, service_times, travel_times


def create_service_times(batch_size, problem_size):
    service_times = numpy.zeros((batch_size, problem_size))
    for x in range(batch_size):
        service_times[x, :] = numpy.random.uniform(0.2, 0.5, problem_size)
    return service_times


def create_duals(batch_size, problem_size):
    duals = numpy.random.uniform(low=0, high=7, size=(batch_size, problem_size))
    return duals


def create_time_matrix(batch_size, problem_size, node_coors, depot_coors):
    time_matrix = numpy.zeros((batch_size, problem_size + 1, problem_size + 1))
    for x in range(batch_size):
        for i in range(problem_size + 1):
            for j in range(problem_size + 1):
                if i != j:
                    if i == 0:
                        time_matrix[x, i, j] = numpy.linalg.norm(depot_coors[x, i, :] - node_coors[x, j - 1, :])
                    elif j == 0:
                        time_matrix[x, i, j] = numpy.linalg.norm(node_coors[x, i - 1, :] - depot_coors[x, j, :])
                    else:
                        time_matrix[x, i, j] = numpy.linalg.norm(node_coors[x, i - 1, :] - node_coors[x, j - 1, :])

    return time_matrix * 2


def create_time_windows(batch_size, problem_size, tw_scalar, minimum_margin=2):
    time_windows = numpy.zeros((batch_size, problem_size, 2), dtype=int)
    for x in range(batch_size):
        for i in range(problem_size):
            time_windows[x, i, 0] = numpy.random.randint(0, 10)
            time_windows[x, i, 1] = numpy.random.randint(time_windows[x, i, 0] + minimum_margin, tw_scalar)
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
