import argparse
import os
import tempfile
import time
import math
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from point import Point
from my_utils import (
    read_data_points,
    read_utility_functions,
    generate_groups,
    generate_fairness_constraints,
    write_results,
    write_results_eld
)
from bigreedy import BiGreedy
from greedy import GreedyAlgorithms
from intcov import run_intcov
from dmm import DMM
from hs import HSAlgorithm
from epskernel import EpsKernel
from bigreedy import BiGreedyPP, BiGreedyPPOptimized

np.random.seed(632)

def calculate_mhr(data_points: List[Point], selected_indices: List[int], utility_funcs: List[List[float]]) -> float:
    if not selected_indices or not utility_funcs:
        return 0.0
    max_regret = 0.0
    for u in utility_funcs:
        utility_selected = max(np.dot(data_points[i].coordinates, u.coordinates) for i in selected_indices)
        utility_max = max(np.dot(p.coordinates, u.coordinates) for p in data_points)
        if utility_max == 0:
            regret = 1.0
        else:
            regret = 1.0 - (utility_selected / utility_max)
        if regret > max_regret:
            max_regret = regret
    return max_regret


def write_aggregated_results(dataset: str, k: int, mhrs: List[float], times: List[float]) -> None:
    result_dir = "./result"
    os.makedirs(result_dir, exist_ok=True)  

    out_file = f"{result_dir}/{dataset[:-4]}_{k}.txt"
    with open(out_file, 'a') as fout:
        if k == 2:
            fout.write(f"{mhrs[0]:8.2f}\t")
        for i in range(1, 10):
            fout.write(f"{mhrs[i]:8.2f}\t")
        fout.write("\n")
        if k == 2:
            fout.write(f"{times[0]:8.2f}\t")
        for i in range(1, 10):
            fout.write(f"{times[i]:8.2f}\t" if mhrs[i] >= 0 else f"{'-1':8}\t")
        fout.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run fairness-aware selection algorithms")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("k", type=int, help="Number of items to select")
    args = parser.parse_args()

    # dataset_path = f"../data/{args.dataset}"
    dataset_path = os.path.join("data", args.dataset)
    # utils_path = f"../utils/utils_{args.dataset.split('_')[0]}d.txt"
    # utils_path = f"../utils/utils_{args.k}d.txt"
    utils_path = os.path.join("utils", f"utils_{args.k}d_10000.txt")
    max_m = 100000
    num_algs = 15

    data_points, dim, group_dim, group_mappings = read_data_points(dataset_path)
    utility_funcs = read_utility_functions(utils_path, max_m)

    for group_id in range(group_dim):
        grouped_data = generate_groups(data_points, group_id)
        fairness_constraints = generate_fairness_constraints(grouped_data, data_points, args.k)

        mhrs = [-1.0] * num_algs
        times = [-1.0] * num_algs

        if (args.k == 2) and len(fairness_constraints) < 7:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                temp_file.write("2\n")
                for p in data_points:
                    # coord = p.coordinates
                    x, y = p.coordinates[0], p.coordinates[1]
                    group = p.get_category(group_id)
                    temp_file.write(f"{x}\t{y}\t{group}\n")
                temp_path = temp_file.name

            upper_bounds = [fc.uc for fc in fairness_constraints.values()]
            lower_bounds = [fc.lc for fc in fairness_constraints.values()]
            total_k = sum(fc.ki for fc in fairness_constraints.values())

            try:
                print("Running IntCov...")
                result, time_taken = run_intcov(temp_path, [2], upper_bounds, lower_bounds, total_k)
                final_result = []
                fair_counts = defaultdict(int)
                for idx in result:
                    if idx < len(data_points):
                        final_result.append(data_points[idx].id)
                        group = data_points[idx].get_category(group_id)
                        fair_counts[group] += 1

                candidate = BiGreedy.construct_candidate(
                    grouped_data, 
                    {k: (v.lc, v.uc, v.ki) for k, v in fairness_constraints.items()},
                    final_result,
                    fair_counts,
                    args.k
                )
                while candidate and len(final_result) < args.k:
                    final_result.append(candidate[0])
                    group = data_points[candidate[0]].get_category(group_id)
                    fair_counts[group] += 1
                    candidate = BiGreedy.construct_candidate(
                        grouped_data,
                        fairness_constraints,
                        final_result,
                        fair_counts,
                        args.k
                    )

                mhrs[0] = calculate_mhr(data_points, final_result, utility_funcs)
                times[0] = time_taken * 1000

                print(f"time taken: {times[0]}\tmhr: {mhrs[0]}\n")

            finally:
                os.remove(temp_path)

        print("Running Bi-Greedy...")
        result, time_alg = BiGreedy.run_bi_greedy(
            grouped_data, fairness_constraints, data_points, group_id, 0.02, utility_funcs, args.k, max_m)
        mhrs[1] = calculate_mhr(data_points, result, utility_funcs)
        times[1] = time_alg
        print(f"time taken: {time_alg}\tmhr: {mhrs[1]}\n")

        print("Running Bi-Greedy-Plus...")
        result, time_alg = BiGreedy.run_bi_greedy_plus(
            grouped_data, fairness_constraints, data_points, group_id, 0.04, 0.02, utility_funcs, args.k, max_m)
        mhrs[2] = calculate_mhr(data_points, result, utility_funcs)
        times[2] = time_alg
        print(f"time taken: {time_alg}\tmhr: {mhrs[2]}\n")

        # Run Bi-GreedyFast++
        # print("Running Bi-Greedy++Fast...")
        # result, time_alg = BiGreedyPPOptimized.run_bi_greedy_pp(
        #     grouped_data, fairness_constraints, data_points, group_id,
        #     0.02, utility_funcs, args.k, max_m=1000, sample_ratio=0.1
        # )

        # mhr = calculate_mhr(data_points, result, utility_funcs)
        # mhrs[4] = mhr  # Assign to a new index (e.g., 15)
        # times[4] = time_alg
        # print(f"time taken: {time_alg}\tmhr: {mhrs[4]}\n")
        
        # Run Bi-Greedy++
        print("Running Bi-Greedy++...")
        result, time_alg = BiGreedyPP.run_bi_greedy_pp(
            grouped_data, fairness_constraints, data_points, group_id,
            0.02, utility_funcs, args.k, max_m=1000, sample_ratio=0.1
        )
        mhr = calculate_mhr(data_points, result, utility_funcs)
        mhrs[3] = mhr  # Assign to a new index (e.g., 15)
        times[3] = time_alg
        print(f"time taken: {time_alg}\tmhr: {mhrs[3]}\n")

        write_aggregated_results(args.dataset, args.k, mhrs, times)

if __name__ == "__main__":
    main()