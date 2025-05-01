from dataclasses import dataclass
import torch

from CVRPTWProblemDef import augment_xy_data_by_8_fold, get_random_problems_from_data


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_tw: torch.Tensor = None
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    ROLLOUT_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    time = None
    # shape: (batch, pomo)


class CVRPTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.rollout_size = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_node_tw = None
        self.saved_service_t = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_tw = None
        # shape: (batch, problem+1)
        self.distance_matrix = None
        # shape: (batch, problem+1, problem+1)
        self.capacity = None
        self.grid_size = None
        self.service_t = None

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.time = None
        # shape: (batch, pomo)
        self.used_vehicles = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy'].float()
        self.saved_node_xy = loaded_dict['node_xy'].float()
        self.saved_node_demand = loaded_dict['node_demand'].float()
        self.grid_size = loaded_dict['grid_size']
        self.capacity = loaded_dict['capacity']
        if 'node_tw' in loaded_dict.keys():
            self.saved_node_tw = loaded_dict['node_tw']
            self.service_t = loaded_dict['service_duration']
        self.saved_index = 0

    def load_problems(self, batch_size, rollout_size, device, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, _, node_tw, self.service_t, capacity = get_random_problems_dummy(batch_size,
                                                                                                             self.problem_size)
            depot_xy = depot_xy.float()
            node_xy = node_xy.float()
            node_demand = node_demand.float()
            self.grid_size = 1000
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size].to(device)
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size].to(device)
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size].to(device)
            capacity = self.capacity[self.saved_index:self.saved_index + batch_size].to(device)

            if self.saved_node_tw is not None:
                node_tw = self.saved_node_tw[self.saved_index:self.saved_index + batch_size].to(device)
            else:
                depot_xy, node_xy, node_demand, _, node_tw, self.service_t = get_random_problems_from_data(
                    depot_xy.to(device), node_xy.to(device), node_demand.to(device))

            self.saved_index += batch_size
            if self.saved_index > self.saved_node_xy.shape[0] - batch_size:
                self.saved_index = 0

        assert node_xy.shape[1] == self.env_params['problem_size']

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy, self.grid_size)
                node_xy = augment_xy_data_by_8_fold(node_xy, self.grid_size)
                node_demand = node_demand.repeat(8, 1)
                capacity = capacity.repeat(8)
                node_tw = node_tw.repeat(8, 1, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        depot_tw = torch.Tensor([0, float('inf')])[None, None].expand(self.batch_size, 1, 2)
        self.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        # shape: (batch, problem+1, 2)
        self.distance_matrix = torch.cdist(self.depot_node_xy, self.depot_node_xy)
        # shape: (batch, problem+1, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size)[None, :].expand(self.batch_size, self.rollout_size)

        # Scale demand to [0, 1] based on vehilce capacity
        self.depot_node_demand /= capacity[:, None]

        # Create neural network input. Scale data
        self.reset_state.depot_xy = depot_xy / self.grid_size
        self.reset_state.node_xy = node_xy / self.grid_size
        self.reset_state.node_demand = self.depot_node_demand[:, 1:]
        self.reset_state.node_tw = node_tw / self.grid_size
        print(node_tw[0, :])

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.rollout_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.rollout_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.rollout_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.rollout_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.time = torch.zeros(size=(self.batch_size, self.rollout_size))
        # shape: (batch, pomo)
        self.used_vehicles = torch.zeros(size=(self.batch_size, self.rollout_size))
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.rollout_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.used_vehicles += self.at_the_depot.int() - self.finished.int()  # Maybe only calculate this after solution is complete?

        # Dynamic-TW
        ####################################
        if self.selected_count == 1:
            travel_time = torch.zeros(self.batch_size, self.rollout_size)
        else:
            travel_time = self._get_travel_distance_last_step()
        tw_start_list = self.depot_node_tw[:, None, :, 0].expand(self.batch_size, self.rollout_size, -1)
        selected_tw_start = torch.gather(tw_start_list, 2, gathering_index).squeeze(dim=2)
        self.time = torch.maximum(self.time + travel_time, selected_tw_start) + self.service_t
        self.time[self.at_the_depot] = 0  # reset time at the depot

        # Compute mask
        ####################################

        self.visited_ninf_flag[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][
            ~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        # Mask nodes that can not be reached before the time window end
        travel_time_to_all = self.distance_matrix[:, None].expand(self.batch_size, self.rollout_size, -1, -1)[
            self.BATCH_IDX, self.ROLLOUT_IDX, selected]
        possible_arrival_time_all = self.time[:, :, None].expand(-1, -1, self.problem_size + 1) + travel_time_to_all
        tw_end_all = self.depot_node_tw[:, None, :, 1].expand(-1, self.rollout_size, -1)
        can_not_be_reached_in_time = possible_arrival_time_all > tw_end_all
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[can_not_be_reached_in_time] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time / self.grid_size  # Scale by grid size

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_total_travel_distance()  # note the minus sign! -self.used_vehicles
        else:
            reward = None

        return self.step_state, reward, done

    def _get_total_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.rollout_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def _get_travel_distance_last_step(self):
        gathering_index = self.selected_node_list[:, :, -2:, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.rollout_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths[:, :, 0]
        # shape: (batch, pomo)
        return travel_distances
