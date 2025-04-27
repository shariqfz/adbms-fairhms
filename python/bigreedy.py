import heapq
import math
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from point import Point
from tqdm import tqdm

class Increment:
    """Priority queue element for max-heap functionality"""
    def __init__(self, index: int, value: float):
        self.index = index
        self.value = value

    def __lt__(self, other) -> bool:
        return self.value > other.value  # Reverse for max-heap

class BiGreedy:
    @staticmethod
    def construct_candidate(grouped_data: Dict[int, List[Point]],
                           fairness_constraints: Dict[int, Tuple[int, int, int]],
                           result: List[int],
                           fair_counts: Dict[int, int],
                           k: int) -> List[int]:
        candidate = []
        remaining = k - len(result)
        
        # Add mandatory candidates from under-represented groups
        for group_id, (lc, uc, ki) in fairness_constraints.items():
            if fair_counts.get(group_id, 0) < lc:
                candidate.extend(p.id for p in grouped_data[group_id] 
                               if p.id not in result)
                remaining -= lc - fair_counts.get(group_id, 0)
                if remaining <= 0:
                    break
        
        # Add optional candidates from valid groups
        if remaining > 0:
            for group_id, (lc, uc, ki) in fairness_constraints.items():
                if lc <= fair_counts.get(group_id, 0) < uc:
                    candidate.extend(p.id for p in grouped_data[group_id]
                                   if p.id not in result)
        
        return list(set(candidate))

    @staticmethod
    def update_priority_queue(data: List[Point],
                             candidate: List[int],
                             utility_funcs: List[Point],
                             max_utility_D: List[Tuple[int, float]],
                             current_utilities: List[Tuple[int, float]],
                             tau: float,
                             epsilon: float) -> List[Increment]:
        heap = []
        for p_id in candidate:
            p = next((x for x in data if x.id == p_id), None)
            if not p:
                continue

            total_delta = 0.0
            for i, (_, max_d) in enumerate(max_utility_D):
                if max_d == 0:
                    continue
                
                current_u = current_utilities[i][1]
                new_u = p.dotP(utility_funcs[i])
                
                new_ratio = min(new_u / max_d, tau)
                old_ratio = min(current_u / max_d, tau)
                total_delta += (new_ratio - old_ratio)

            if total_delta > 0:
                heapq.heappush(heap, Increment(p_id, total_delta))
        
        return heap

    @staticmethod
    def multiround_greedy(grouped_data: Dict[int, List[Point]],
                         fairness_constraints: Dict[int, Tuple[int, int, int]],
                         data: List[Point],
                         group_id: int,
                         utility_funcs: List[Point],
                         k: int,
                         m: int,
                         gamma: int,
                         epsilon: float,
                         tau: float) -> Tuple[List[int], Dict[int, int]]:
        result = []
        fair_counts = defaultdict(int)
        max_utility_D = []
        
        # Initialize max utilities for dataset
        for uf in tqdm(utility_funcs[:m], desc="Max Utility"):
            max_u = -float('inf')
            max_id = -1
            for p in data:
                current_u = p.dotP(uf)
                if current_u > max_u:
                    max_u = current_u
                    max_id = p.id
            max_utility_D.append((max_id, max_u))

        current_utilities = [(-1, 0.0) for _ in range(m)]
        
        for _ in tqdm(range(gamma), desc="Multi-round Greedy; gamma"):
            candidate = BiGreedy.construct_candidate(
                grouped_data, fairness_constraints, result, fair_counts, k)
            if not candidate:
                break

            heap = BiGreedy.update_priority_queue(
                data, candidate, utility_funcs[:m], 
                max_utility_D, current_utilities, tau, epsilon)

            while heap and len(result) < k:
                item = heapq.heappop(heap)
                p = next((x for x in data if x.id == item.index), None)
                if not p or item.index in result:
                    continue

                # Update current utilities
                new_utilities = []
                for i, (_, current_u) in enumerate(current_utilities):
                    new_u = p.dotP(utility_funcs[i])
                    new_utilities.append((p.id, max(new_u, current_u)))

                # Check group constraints
                group = p.get_category(group_id)
                if fair_counts[group] >= fairness_constraints[group][1]:
                    continue

                # Add to result
                result.append(p.id)
                fair_counts[group] += 1
                current_utilities = new_utilities

                # Update candidate and queue
                candidate = BiGreedy.construct_candidate(
                    grouped_data, fairness_constraints, result, fair_counts, k)
                heap = BiGreedy.update_priority_queue(
                    data, candidate, utility_funcs[:m], 
                    max_utility_D, current_utilities, tau, epsilon)

        return result, fair_counts

    @staticmethod
    def bi_search(grouped_data: Dict[int, List[Point]],
                 fairness_constraints: Dict[int, Tuple[int, int, int]],
                 data: List[Point],
                 group_id: int,
                 utility_funcs: List[Point],
                 k: int,
                 m: int,
                 epsilon: float,
                 gamma: int) -> Tuple[List[int], float]:
        tau_low, tau_high = 0.0, 1.0
        best_result = []
        best_fair_counts = defaultdict(int)

        for _ in tqdm(range(int(math.log2(1/epsilon)) + 1), desc="Binary Search"):
            tau = (tau_low + tau_high) / 2
            result, fair_counts = BiGreedy.multiround_greedy(
                grouped_data, fairness_constraints, data, group_id,
                utility_funcs, k, m, gamma, epsilon, tau)

            if len(result) >= k:
                best_result = result
                best_fair_counts = fair_counts
                tau_low = tau
            else:
                tau_high = tau

        return best_result, tau_low

    @staticmethod
    def run_bi_greedy(grouped_data: Dict[int, List[Point]],
                     fairness_constraints: Dict[int, Tuple[int, int, int]],
                     data: List[Point],
                     group_id: int,
                     epsilon: float,
                     utility_funcs: List[Point],
                     k: int,
                     max_m: int = 1000) -> Tuple[List[int], float]:
        start_time = time.time()
        m = min(len(utility_funcs), max_m)
        gamma = math.ceil(math.log(2 * m / epsilon))

        result, _ = BiGreedy.bi_search(
            grouped_data, fairness_constraints, data, group_id,
            utility_funcs, k, m, epsilon, gamma)

        # Fill remaining slots
        fair_counts = defaultdict(int)
        for p_id in result:
            p = next(x for x in data if x.id == p_id)
            fair_counts[p.get_category(group_id)] += 1

        while len(result) < k:
            candidate = BiGreedy.construct_candidate(
                grouped_data, fairness_constraints, result, fair_counts, k)
            if not candidate:
                break
            result.append(candidate[0])
            p = next(x for x in data if x.id == candidate[0])
            fair_counts[p.get_category(group_id)] += 1

        return result, time.time() - start_time

    @staticmethod
    def run_bi_greedy_plus(grouped_data: Dict[int, List[Point]],
                          fairness_constraints: Dict[int, Tuple[int, int, int]],
                          data: List[Point],
                          group_id: int,
                          lambda_val: float,
                          epsilon: float,
                          utility_funcs: List[Point],
                          k: int,
                          max_m: int = 1000) -> Tuple[List[int], float]:
        start_time = time.time()
        m = min(len(utility_funcs), max_m)
        gamma = math.ceil(math.log(2 * m / epsilon))

        # Initial search
        result, tau = BiGreedy.bi_search(
            grouped_data, fairness_constraints, data, group_id,
            utility_funcs, k, m, epsilon, gamma)

        # Iterative refinement
        while m < max_m:
            prev_m = m
            m = min(2 * m, max_m)
            if m == prev_m:
                break

            # Update utility functions
            new_utility = utility_funcs[:m]
            new_result, new_tau = BiGreedy.bi_search(
                grouped_data, fairness_constraints, data, group_id,
                new_utility, k, m, epsilon, gamma)

            if len(new_result) >= k and new_tau >= tau - lambda_val:
                result = new_result
                tau = new_tau
            else:
                break

        return result, time.time() - start_time

    @staticmethod
    def run_bi_greedy_with_delta(grouped_data: Dict[int, List[Point]],
                                fairness_constraints: Dict[int, Tuple[int, int, int]],
                                data: List[Point],
                                group_id: int,
                                epsilon: float,
                                utility_funcs: List[Point],
                                k: int,
                                max_m: int,
                                deltaC: float) -> Tuple[List[int], float]:
        start_time = time.time()
        m = min(int(k * data[0].dim * deltaC), max_m)
        gamma = math.ceil(math.log(2 * m / epsilon))

        result, _ = BiGreedy.bi_search(
            grouped_data, fairness_constraints, data, group_id,
            utility_funcs[:m], k, m, epsilon, gamma)

        # Post-processing
        fair_counts = defaultdict(int)
        for p_id in result:
            p = next(x for x in data if x.id == p_id)
            fair_counts[p.get_category(group_id)] += 1

        candidate = BiGreedy.construct_candidate(
            grouped_data, fairness_constraints, result, fair_counts, k)
        while candidate and len(result) < k:
            result.append(candidate[0])
            p = next(x for x in data if x.id == candidate[0])
            fair_counts[p.get_category(group_id)] += 1
            candidate = BiGreedy.construct_candidate(
                grouped_data, fairness_constraints, result, fair_counts, k)

        return result, time.time() - start_time

    @staticmethod
    def run_bi_greedy_plus_with_delta(grouped_data: Dict[int, List[Point]],
                                     fairness_constraints: Dict[int, Tuple[int, int, int]],
                                     data: List[Point],
                                     group_id: int,
                                     lambda_val: float,
                                     epsilon: float,
                                     utility_funcs: List[Point],
                                     k: int,
                                     max_m: int,
                                     deltaC: float) -> Tuple[List[int], float]:
        start_time = time.time()
        m = min(int(k * data[0].dim * deltaC), max_m)
        gamma = math.ceil(math.log(2 * m / epsilon))

        # Initial search
        result, tau = BiGreedy.bi_search(
            grouped_data, fairness_constraints, data, group_id,
            utility_funcs[:m], k, m, epsilon, gamma)

        # Multi-phase refinement
        phases = 2 if deltaC == 0.25 else 1
        for _ in range(phases):
            prev_m = m
            m = min(m * 2, max_m)
            if m == prev_m:
                break

            new_result, new_tau = BiGreedy.bi_search(
                grouped_data, fairness_constraints, data, group_id,
                utility_funcs[:m], k, m, epsilon, gamma)

            if len(new_result) >= k and new_tau >= max(tau - lambda_val, 0.001):
                result = new_result
                tau = new_tau
            else:
                break

        return result, time.time() - start_time