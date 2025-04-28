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
        sum_delta = 0.0
        
        # Initialize max utilities for dataset
        max_utility_D = []
        for uf in utility_funcs[:m]:
            max_u = -float('inf')
            max_id = -1
            for p in data:
                current_u = p.dotP(uf)
                if current_u > max_u:
                    max_u = current_u
                    max_id = p.id
            max_utility_D.append((max_id, max_u))

        current_utilities = [(-1, 0.0) for _ in range(m)]
        
        for _ in tqdm(range(gamma), desc="Multi-round Greedy"):
            candidate = BiGreedy.construct_candidate(
                grouped_data, fairness_constraints, result, fair_counts, k)
            if not candidate:
                break

            heap = BiGreedy.update_priority_queue(
                data, candidate, utility_funcs[:m], 
                max_utility_D, current_utilities, tau, epsilon)

            tmp_result = []
            while heap and len(result) < k and len(tmp_result) < k:
                item = heapq.heappop(heap)
                p = next((x for x in data if x.id == item.index), None)
                if not p or p.id in result:
                    continue

                # Calculate delta for this point
                total_delta = 0.0
                new_utilities = []
                for i, (_, current_u) in enumerate(current_utilities):
                    new_u = p.dotP(utility_funcs[i])
                    if max_utility_D[i][1] == 0:
                        continue  # Avoid division by zero
                    new_ratio = min(new_u / max_utility_D[i][1], tau)
                    old_ratio = min(current_u / max_utility_D[i][1], tau)
                    total_delta += (new_ratio - old_ratio)
                    new_utilities.append((p.id, max(new_u, current_u)))

                # Check group constraints
                group = p.get_category(group_id)
                if fair_counts[group] >= fairness_constraints[group][1]:
                    continue

                # Add to result
                result.append(p.id)
                tmp_result.append(p.id)
                fair_counts[group] += 1
                sum_delta += total_delta / m  # Average delta per utility
                current_utilities = new_utilities

                # Update candidate and queue
                candidate = BiGreedy.construct_candidate(
                    grouped_data, fairness_constraints, result, fair_counts, k)
                heap = BiGreedy.update_priority_queue(
                    data, candidate, utility_funcs[:m], 
                    max_utility_D, current_utilities, tau, epsilon)

            # Update max_utility_D for remaining points after the round
            if tmp_result:
                for i in range(m):
                    max_u = -float('inf')
                    max_id = -1
                    for p in data:
                        if p.id in result:
                            continue
                        current_u = p.dotP(utility_funcs[i])
                        if current_u > max_u:
                            max_u = current_u
                            max_id = p.id
                    max_utility_D[i] = (max_id, max_u)

        # Clear results if sum_delta is insufficient
        threshold = (1 - epsilon / (2 * m)) * tau
        if sum_delta < threshold:
            result.clear()
            fair_counts.clear()

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

        while tau_high - tau_low >= epsilon:
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
        dim = data[0].dimension if data else 0
        m = min(k * dim * 10, max_m, len(utility_funcs))  # Match C++ logic
        gamma = math.ceil(math.log2(2 * m / epsilon))  # Base-2 logarithm

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
        dim = data[0].dimension if data else 0
        m = min(dim * k * 2, max_m, len(utility_funcs))  # Match C++: m = dim*k*2
        gamma = math.ceil(math.log2(2 * m / epsilon))  # Base-2 logarithm

        # Initialize max_utility_D for initial m utilities
        max_utility_D = []
        for i in range(m):
            max_u = -float('inf')
            max_id = -1
            for p in data:
                current_u = p.dotP(utility_funcs[i])
                if current_u > max_u:
                    max_u = current_u
                    max_id = p.id
            max_utility_D.append((max_id, max_u))

        # Initial binary search
        result, tau = BiGreedy.bi_search(
            grouped_data, fairness_constraints, data, group_id,
            utility_funcs, k, m, epsilon, gamma
        )

        # Multi-phase refinement (C++ loop behavior)
        while True:
            prev_m = m
            m = min(m * 2, max_m, len(utility_funcs))
            if m == prev_m:
                break

            # Extend max_utility_D for new utilities
            for i in range(prev_m, m):
                max_u = -float('inf')
                max_id = -1
                for p in data:
                    current_u = p.dotP(utility_funcs[i])
                    if current_u > max_u:
                        max_u = current_u
                        max_id = p.id
                max_utility_D.append((max_id, max_u))

            # Run multiround greedy with tau - lambda
            tmp_result, tmp_fair_counts = BiGreedy.multiround_greedy(
                grouped_data, fairness_constraints, data, group_id,
                utility_funcs, k, m, gamma, epsilon, tau - lambda_val
            )

            if tmp_result:
                # Update result and tau with new binary search
                new_result, new_tau = BiGreedy.bi_search(
                    grouped_data, fairness_constraints, data, group_id,
                    utility_funcs, k, m, epsilon, gamma
                )
                if len(new_result) >= k and new_tau >= tau - lambda_val:
                    result = new_result
                    tau = new_tau
                else:
                    break
            else:
                # Adjust tau bounds if no result
                tau_high = tau - lambda_val
                tau_low = 0.0
                adj_result, adj_tau = BiGreedy.bi_search(
                    grouped_data, fairness_constraints, data, group_id,
                    utility_funcs, k, m, epsilon, gamma
                )
                if len(adj_result) >= k:
                    result = adj_result
                    tau = adj_tau
                else:
                    break

        # Post-processing: Fill remaining slots
        fair_counts = defaultdict(int)
        for p_id in result:
            p = next(x for x in data if x.id == p_id)
            fair_counts[p.get_category(group_id)] += 1

        while len(result) < k:
            candidate = BiGreedy.construct_candidate(
                grouped_data, fairness_constraints, result, fair_counts, k
            )
            if not candidate:
                break
            result.append(candidate[0])
            p = next(x for x in data if x.id == candidate[0])
            fair_counts[p.get_category(group_id)] += 1

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
    

import heapq
import math
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from point import Point

class BiGreedyPP:
    @staticmethod
    def lazy_greedy(
        data: List[Point],
        utility_funcs: List[Point],
        grouped_data: Dict[int, List[Point]],
        fairness_constraints: Dict[int, Tuple[int, int, int]],
        k: int,
        m: int,
        gamma: int,
        epsilon: float,
        tau: float,
        group_id: int,  # Added parameter
        sample_ratio: float = 0.1
    ) -> List[int]:
        """Lazy stochastic greedy algorithm with fairness constraints."""
        result = []
        fair_counts = defaultdict(int)
        candidate_set = [p.id for p in data]
        heap = []
        
        # Initialize marginal gains with a random subset
        subset = np.random.choice(candidate_set, size=int(len(candidate_set)*sample_ratio), replace=False)
        for p_id in subset:
            p = next(x for x in data if x.id == p_id)
            group = p.get_category(group_id)  # Now group_id is defined
            if fair_counts[group] >= fairness_constraints[group][1]:
                continue
            current_gain = BiGreedyPP.marginal_gain(p_id, result, utility_funcs, data, tau)
            heapq.heappush(heap, (-current_gain, p_id))  # Max-heap
        
        while len(result) < k and heap:
            _, p_id = heapq.heappop(heap)
            p = next(x for x in data if x.id == p_id)
            group = p.get_category(group_id)  # group_id is defined
            # Check fairness constraint
            if fair_counts[group] >= fairness_constraints[group][1]:
                continue
            
            # Re-evaluate marginal gain if not in the initial sample
            if p_id not in subset:
                current_gain = BiGreedyPP.marginal_gain(p_id, result, utility_funcs, data, tau)
                heapq.heappush(heap, (-current_gain, p_id))
            else:
                result.append(p_id)
                fair_counts[group] += 1
                # Update candidate pool with new random sample
                subset = np.random.choice([x for x in candidate_set if x not in result], 
                                         size=int(len(candidate_set)*sample_ratio), replace=False)
                for new_id in subset:
                    new_p = next(x for x in data if x.id == new_id)
                    new_group = new_p.get_category(group_id)  # group_id is defined
                    if fair_counts[new_group] >= fairness_constraints[new_group][1]:
                        continue
                    gain = BiGreedyPP.marginal_gain(new_id, result, utility_funcs, data, tau)
                    heapq.heappush(heap, (-gain, new_id))
        return result

    @staticmethod
    def marginal_gain(p_id: int, current_set: List[int], 
                     utility_funcs: List[Point], data: List[Point], 
                     tau: float) -> float:
        """Compute marginal gain of adding p_id to current_set."""
        p = next(x for x in data if x.id == p_id)
        min_hr = np.inf
        for u in utility_funcs:
            current_max = max((x.dotP(u) for x_id in current_set for x in data if x.id == x_id), default=0)
            new_hr = min(p.dotP(u) / u.dotP(u), tau)  # Truncated ratio
            if new_hr > current_max:
                gain = new_hr - current_max
                if gain < min_hr:
                    min_hr = gain
        return min_hr

    @staticmethod
    def hierarchical_sampling(dim: int, initial_samples: int = 100, max_depth: int = 3) -> List[Point]:
        """Hierarchical sampling of utility vectors."""
        samples = []
        # Step 1: Uniform initial sampling
        samples.extend([Point.random_sphere(dim) for _ in range(initial_samples)])
        
        for _ in range(max_depth):
            # Step 2: Perturb existing vectors
            new_samples = []
            for u in samples:
                perturbation = np.random.normal(0, 0.1, dim)
                new_coords = u.coordinates + perturbation
                new_coords /= np.linalg.norm(new_coords)  # Normalize
                new_samples.append(Point(dimension=dim, coordinates=new_coords))
            samples.extend(new_samples)
        return samples

    @staticmethod
    def run_bi_greedy_pp(
        grouped_data: Dict[int, List[Point]],
        fairness_constraints: Dict[int, Tuple[int, int, int]],
        data: List[Point],
        group_id: int,
        epsilon: float,
        utility_funcs: List[Point],
        k: int,
        max_m: int = 1000,
        sample_ratio: float = 0.1
    ) -> Tuple[List[int], float]:
        start_time = time.time()
        dim = data[0].dimension if data else 0
        
        # Hierarchical utility sampling
        utility_samples = BiGreedyPP.hierarchical_sampling(dim)
        m = len(utility_samples)
        
        # Multi-round lazy greedy search
        gamma = math.ceil(math.log2(2 * m / epsilon))
        result, tau = BiGreedy.bi_search(
            grouped_data, fairness_constraints, data, group_id,
            utility_samples, k, m, epsilon, gamma
        )
        
        # Refinement with lazy stochastic greedy
        refined_result = BiGreedyPP.lazy_greedy(
            data, utility_samples, grouped_data, fairness_constraints,
            k, m, gamma, epsilon, tau, sample_ratio
        )
        
        # Post-processing to ensure fairness
        fair_counts = defaultdict(int)
        for p_id in refined_result:
            p = next(x for x in data if x.id == p_id)
            fair_counts[p.get_category(group_id)] += 1
        
        while len(refined_result) < k:
            candidate = BiGreedy.construct_candidate(
                grouped_data, fairness_constraints, refined_result, fair_counts, k
            )
            if not candidate:
                break
            refined_result.append(candidate[0])
        
        return refined_result, time.time() - start_time