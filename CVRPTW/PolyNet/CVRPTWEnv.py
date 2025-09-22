from dataclasses import dataclass
import torch

from CVRPTWProblemDef import augment_xy_data_by_8_fold, get_random_problems


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
    node_st: torch.Tensor = None
    # shape: (batch, problem)


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
        self.depot_node_st = None
        self.depot_node_tw = None
        # shape: (batch, problem+1)
        self.travel_times = None
        # shape: (batch, problem+1, problem+1)

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
        self.total_travel_time = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def declare_problem(self, depot_xy, node_xy, node_demand, node_tw, node_st,
                    travel_times, batch_size):

        self.batch_size = batch_size

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_tw = node_tw
        self.reset_state.node_st = node_st

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        depot_tw = torch.Tensor([0, 1])[None, None].expand(self.batch_size, 1, 2)
        self.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        depot_st = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_st = torch.cat((depot_st, node_st), dim=1)
        self.travel_times = travel_times

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def load_problems(self, batch_size, rollout_size, device, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size


        depot_xy, node_xy, node_demand, node_tw, node_st, travel_times = get_random_problems(batch_size,
                                                                                                         self.problem_size)
        self.travel_times = travel_times

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_tw = node_tw.repeat(8, 1, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        depot_st = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_st = torch.cat((depot_st,node_st),dim=1)
        depot_tw = torch.Tensor([0, 1])[None, None].expand(self.batch_size, 1, 2)
        self.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        # shape: (batch, problem+1, 2)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size)[None, :].expand(self.batch_size, self.rollout_size)


        # Create neural network input. Scale data
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_tw = node_tw
        self.reset_state.node_st = node_st

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
        self.total_travel_time = torch.zeros(size=(self.batch_size, self.rollout_size))

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
        if self.current_node is not None:
            previous_indexes = self.current_node
            previous_indexes = previous_indexes[:, :, None].expand(-1, -1, self.problem_size + 1)
            previous_indexes = previous_indexes[:, :, None, :]

        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.rollout_size, -1)
        st_list =  self.depot_node_st[:, None, :].expand(self.batch_size, self.rollout_size, -1)
        # shape: (batch, pomo, problem+1)
        travel_time_list = self.travel_times[:, None, :, :].expand(self.batch_size, self.rollout_size, -1, -1)
        # shape: (batch, pomo, problem+1, problem+1)

        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        selected_st = st_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        if self.selected_count == 1:
            travel_times = torch.zeros(self.batch_size, self.rollout_size)
        else:
            selected_travel_matrices = travel_time_list.gather(dim=2, index=previous_indexes).squeeze(dim=2)
            # shape: (batch, pomo, problem+1)
            travel_times = selected_travel_matrices.gather(dim=2, index=gathering_index).squeeze(dim=2)
            # shape: (batch, pomo)
            self.total_travel_time += travel_times

        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        # Dynamic-TW
        ####################################

        tw_start_list = self.depot_node_tw[:, None, :, 0].expand(self.batch_size, self.rollout_size, -1)
        selected_tw_start = torch.gather(tw_start_list, 2, gathering_index).squeeze(dim=2)
        self.time = torch.maximum(self.time + travel_times, selected_tw_start) + selected_st
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
        travel_time_to_all = self.travel_times[:, None].expand(self.batch_size, self.rollout_size, -1, -1)[
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
        self.step_state.time = self.time

        # returning values
        done = self.finished.all()
        if done:
            reward = -self.total_travel_time  # note the minus sign! -self.used_vehicles
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
