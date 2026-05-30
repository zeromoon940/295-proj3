import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))


@dataclass
class InstanceData:
    map_name: str
    scenario_id: int
    agent_count: int
    grid: list
    starts: list
    goals: list
    shortest_lengths: list
    manhattan_lengths: list
    shortest_paths: list
    shortest_sets: list
    height: int
    width: int


class ReservationTable:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.final_times = {}
        self.latest_time = 0

    def add_path(self, path):
        for timestep, loc in enumerate(path):
            self.vertices.setdefault(timestep, set()).add(loc)
            self.latest_time = max(self.latest_time, timestep)
        for timestep in range(1, len(path)):
            self.edges.setdefault(timestep, set()).add((path[timestep], path[timestep - 1]))
            self.latest_time = max(self.latest_time, timestep)
        goal_time = len(path) - 1
        loc = path[-1]
        current = self.final_times.get(loc)
        if current is None or goal_time < current:
            self.final_times[loc] = goal_time
        self.latest_time = max(self.latest_time, goal_time)

    def blocked(self, curr_loc, next_loc, next_time):
        if next_loc in self.vertices.get(next_time, ()):
            return True
        if (curr_loc, next_loc) in self.edges.get(next_time, ()):
            return True
        final_time = self.final_times.get(next_loc)
        if final_time is not None and next_time >= final_time:
            return True
        return False

    def future_blocked(self, loc, time_step):
        final_time = self.final_times.get(loc)
        if final_time is not None and final_time >= time_step:
            return True
        for timestep, locs in self.vertices.items():
            if timestep > time_step and loc in locs:
                return True
        return False


def read_map(path):
    lines = Path(path).read_text().splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    start = lines.index("map") + 1
    rows = lines[start:start + height]
    grid = []
    for row in rows:
        grid.append([cell in ("@", "T") for cell in row[:width]])
    return grid


def read_scenario(path, limit=None):
    rows = []
    for raw in Path(path).read_text().splitlines():
        if not raw or raw.startswith("version"):
            continue
        parts = raw.split()
        start = (int(parts[5]), int(parts[4]))
        goal = (int(parts[7]), int(parts[6]))
        rows.append((start, goal))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def passable(grid, loc):
    row, col = loc
    return 0 <= row < len(grid) and 0 <= col < len(grid[0]) and not grid[row][col]


def compute_heuristics(grid, goal):
    open_list = deque([goal])
    distances = {goal: 0}
    while open_list:
        row, col = open_list.popleft()
        next_cost = distances[(row, col)] + 1
        for drow, dcol in MOVES[:4]:
            nxt = (row + drow, col + dcol)
            if passable(grid, nxt) and nxt not in distances:
                distances[nxt] = next_cost
                open_list.append(nxt)
    return distances


def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node[2])
        node = node[5]
    path.reverse()
    return path


def astar_reserved(grid, start, goal, h_values, reservation):
    if start not in h_values or reservation.blocked(start, start, 0):
        return None
    rows = len(grid)
    cols = len(grid[0])
    max_time = reservation.latest_time + h_values[start] + rows + cols + 10
    max_expansions = max(12000, rows * cols * 2)
    root = (h_values[start], h_values[start], start, 0, 0, None)
    open_list = [root]
    closed = {(start, 0): 0}
    expansions = 0
    while open_list:
        node = heapq.heappop(open_list)
        expansions += 1
        if expansions > max_expansions:
            return None
        _, _, loc, g_val, timestep, parent = node
        if loc == goal and not reservation.future_blocked(loc, timestep):
            return reconstruct_path(node)
        if timestep >= max_time:
            continue
        for drow, dcol in MOVES:
            nxt = (loc[0] + drow, loc[1] + dcol)
            if not passable(grid, nxt) or nxt not in h_values:
                continue
            next_time = timestep + 1
            if reservation.blocked(loc, nxt, next_time):
                continue
            next_cost = g_val + 1
            key = (nxt, next_time)
            old_cost = closed.get(key)
            if old_cost is not None and old_cost <= next_cost:
                continue
            closed[key] = next_cost
            h_val = h_values[nxt]
            child = (next_cost + h_val, h_val, nxt, next_cost, next_time, node)
            heapq.heappush(open_list, child)
    return None


def shortest_path_from_heuristics(start, goal, h_values):
    if start not in h_values:
        return None
    loc = start
    path = [loc]
    while loc != goal:
        current = h_values[loc]
        candidates = []
        for drow, dcol in MOVES[:4]:
            nxt = (loc[0] + drow, loc[1] + dcol)
            if nxt in h_values and h_values[nxt] == current - 1:
                candidates.append(nxt)
        if not candidates:
            return None
        loc = sorted(candidates)[0]
        path.append(loc)
    return path


def plan_prioritized(instance, order):
    started = time.perf_counter()
    reservation = ReservationTable()
    paths = [None for _ in instance.starts]
    for agent in order:
        path = astar_reserved(instance.grid, instance.starts[agent], instance.goals[agent], instance.heuristics[agent], reservation)
        if path is None:
            return {"success": False, "cost": None, "runtime": time.perf_counter() - started, "paths": None}
        paths[agent] = path
        reservation.add_path(path)
    return {"success": validate_paths(paths), "cost": sum(len(path) - 1 for path in paths), "runtime": time.perf_counter() - started, "paths": paths}


def validate_paths(paths):
    if any(path is None for path in paths):
        return False
    horizon = max(len(path) for path in paths)
    for timestep in range(horizon):
        seen = {}
        for agent, path in enumerate(paths):
            loc = path[timestep] if timestep < len(path) else path[-1]
            if loc in seen:
                return False
            seen[loc] = agent
        if timestep == 0:
            continue
        edges = set()
        for path in paths:
            prev = path[timestep - 1] if timestep - 1 < len(path) else path[-1]
            curr = path[timestep] if timestep < len(path) else path[-1]
            if (curr, prev) in edges:
                return False
            edges.add((prev, curr))
    return True


def prepare_instance(map_name, scenario_id, agent_count, grid, scenario_rows):
    starts = [row[0] for row in scenario_rows[:agent_count]]
    goals = [row[1] for row in scenario_rows[:agent_count]]
    heuristics = []
    shortest_lengths = []
    manhattan_lengths = []
    shortest_paths = []
    shortest_sets = []
    for start, goal in zip(starts, goals):
        h_values = compute_heuristics(grid, goal)
        heuristics.append(h_values)
        length = h_values.get(start)
        path = shortest_path_from_heuristics(start, goal, h_values)
        shortest_lengths.append(length if length is not None else math.inf)
        manhattan_lengths.append(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
        shortest_paths.append(path)
        shortest_sets.append(set(path) if path is not None else set())
    instance = InstanceData(map_name, scenario_id, agent_count, grid, starts, goals, shortest_lengths, manhattan_lengths, shortest_paths, shortest_sets, len(grid), len(grid[0]))
    instance.heuristics = heuristics
    return instance


def pair_features(instance, i, j):
    scale = max(instance.height, instance.width)
    li = instance.shortest_lengths[i] / scale
    lj = instance.shortest_lengths[j] / scale
    mi = instance.manhattan_lengths[i] / scale
    mj = instance.manhattan_lengths[j] / scale
    si = instance.starts[i]
    sj = instance.starts[j]
    gi = instance.goals[i]
    gj = instance.goals[j]
    shared = len(instance.shortest_sets[i] & instance.shortest_sets[j])
    min_len = max(1, min(len(instance.shortest_sets[i]), len(instance.shortest_sets[j])))
    union_len = max(1, len(instance.shortest_sets[i] | instance.shortest_sets[j]))
    start_dist = (abs(si[0] - sj[0]) + abs(si[1] - sj[1])) / scale
    goal_dist = (abs(gi[0] - gj[0]) + abs(gi[1] - gj[1])) / scale
    cross_a = (abs(si[0] - gj[0]) + abs(si[1] - gj[1])) / scale
    cross_b = (abs(sj[0] - gi[0]) + abs(sj[1] - gi[1])) / scale
    return np.array([
        li,
        lj,
        li - lj,
        mi,
        mj,
        mi - mj,
        shared / scale,
        shared / min_len,
        shared / union_len,
        1.0 if shared > 0 else 0.0,
        start_dist,
        goal_dist,
        cross_a,
        cross_b,
        cross_a - cross_b,
        si[0] / instance.height,
        si[1] / instance.width,
        gi[0] / instance.height,
        gi[1] / instance.width,
        sj[0] / instance.height,
        sj[1] / instance.width,
        gj[0] / instance.height,
        gj[1] / instance.width,
    ], dtype=np.float32)


def baseline_order(instance, name, rng):
    agents = list(range(instance.agent_count))
    if name == "random":
        rng.shuffle(agents)
        return agents
    if name == "shortest":
        return sorted(agents, key=lambda a: (-instance.shortest_lengths[a], a))
    if name == "manhattan":
        return sorted(agents, key=lambda a: (-instance.manhattan_lengths[a], a))
    if name == "natural":
        return agents
    raise ValueError(name)
