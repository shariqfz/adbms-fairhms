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
from bigreedy import BiGreedyPP  

np.random.seed(632)

class Algorithms:
    @staticmethod
    def find_point_indices(selected_points: List[Point], data: List[Point]) -> List[int]:
        selected_indices = []
        for p in selected_points:
            for i, data_p in enumerate(data):
                if np.array_equal(data_p.coordinates, p.coordinates):
                    selected_indices.append(i)
                    break
        return selected_indices

    @staticmethod
    def run_dmmrrms(data: List[Point], k: int) -> Tuple[List[int], float]:
        if not data:
            return [], 0.0
        selected_points, time_elapsed = DMM.run_dmm_rrms(data, r=k, k=0)
        return Algorithms.find_point_indices(selected_points, data), time_elapsed

    @staticmethod
    def run_epskernel(data: List[Point], k: int) -> Tuple[List[int], float]:
        if not data:
            return [], 0.0
        selected_points, time_elapsed = EpsKernel.run_eps_kernel(data, r=k, k=k, curSky=[])
        return Algorithms.find_point_indices(selected_points, data), time_elapsed

    @staticmethod
    def run_hs(data: List[Point], k: int) -> Tuple[List[int], float]:
        if not data:
            return [], 0.0
        selected_points, time_elapsed = HSAlgorithm.run_hs(data, r=k, k=1, curSky=data)
        return Algorithms.find_point_indices(selected_points, data), time_elapsed

    @staticmethod
    def run_sphere(data: List[Point], k: int) -> Tuple[List[int], float]:
        return [], 0.0

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

def run_baselines(data_points: List[Point], grouped_data: Dict[int, List[Point]],
                fairness_constraints: Dict[int, Any], group_id: int,
                k: int, mhrs: List[float], times: List[float]) -> None:
    # RDP-Greedy on each group
    print("Running Greedy on each group...")
    time_total = 0.0
    result_greedy = []
    for group in grouped_data:
        curP = grouped_data[group]
        r = fairness_constraints[group].ki
        selected, time_alg = GreedyAlgorithms.run_greedy(curP, r)
        time_total += time_alg
        result_greedy.extend(Algorithms.find_point_indices(selected, data_points))
    mhr = calculate_mhr(data_points, result_greedy, [])
    mhrs[3] = mhr
    times[3] = time_total
    print(f"time taken: {time_alg}\tmhr: {mhrs[3]}\n")

    print("Running Greedy directly...")
    # RDP-Greedy directly
    selected, time_alg = GreedyAlgorithms.run_greedy(data_points, k)
    result_greedy = Algorithms.find_point_indices(selected, data_points)
    mhr = calculate_mhr(data_points, result_greedy, [])
    mhrs[13] = mhr
    times[13] = time_alg
    print(f"time taken: {time_alg}\tmhr: {mhrs[13]}\n")

    # Fair RDP-Greedy (MRDP)
    print("Running Matroid-Greedy...")
    selected, time_alg = GreedyAlgorithms.run_matroid_greedy(data_points, k, group_id, grouped_data, fairness_constraints)
    result_greedy = Algorithms.find_point_indices(selected, data_points)
    mhr = calculate_mhr(data_points, result_greedy, [])
    mhrs[8] = mhr
    times[8] = time_alg if mhr >= 0 else -1
    print(f"time taken: {time_alg}\tmhr: {mhrs[8]}\n")

    # DMM-RRMS on each group
    time_total = 0.0
    result_dmm = []
    for group in grouped_data:
        curP = grouped_data[group]
        r = fairness_constraints[group].ki
        if r < len(curP[0].coordinates) if curP else 0:
            continue
        selected, time_alg = DMM.run_dmm_rrms(curP, r)
        time_total += time_alg
        result_dmm.extend(Algorithms.find_point_indices(selected, data_points))
    mhr = calculate_mhr(data_points, result_dmm, [])
    mhrs[4] = mhr
    times[4] = time_total
    print(f"time taken: {time_alg}\tmhr: {mhrs[4]}\n")

    # DMM-RRMS directly
    selected, time_alg = DMM.run_dmm_rrms(data_points, k)
    result_dmm = Algorithms.find_point_indices(selected, data_points)
    mhr = calculate_mhr(data_points, result_dmm, [])
    mhrs[14] = mhr
    times[14] = time_alg
    print(f"time taken: {time_alg}\tmhr: {mhrs[14]}\n")

    # Eps-Kernel on each group
    # time_total = 0.0
    # result_eps = []
    # for group in grouped_data:
    #     curP = grouped_data[group]
    #     r = fairness_constraints[group].ki
    #     selected, time_alg = EpsKernel.run_eps_kernel(curP, r)
    #     time_total += time_alg
    #     result_eps.extend(Algorithms.find_point_indices(selected, data_points))
    # mhr = calculate_mhr(data_points, result_eps, [])
    # mhrs[6] = mhr
    # times[6] = time_total
    # print(f"time taken: {time_alg}\tmhr: {mhrs[6]}\n")

    # # Eps-Kernel directly
    # selected, time_alg = EpsKernel.run_eps_kernel(data_points, k)
    # result_eps = Algorithms.find_point_indices(selected, data_points)
    # mhr = calculate_mhr(data_points, result_eps, [])
    # mhrs[11] = mhr
    # times[11] = time_alg
    # print(f"time taken: {time_alg}\tmhr: {mhrs[11]}\n")

    # HS on each group
    time_total = 0.0
    result_hs = []
    for group in grouped_data:
        curP = grouped_data[group]
        r = fairness_constraints[group].ki
        selected, time_alg = HSAlgorithm.run_hs(dataP=curP, r=r, k=1, curSky=[])
        time_total += time_alg
        result_hs.extend(Algorithms.find_point_indices(selected, data_points))
    mhr = calculate_mhr(data_points, result_hs, [])
    mhrs[7] = mhr
    times[7] = time_total
    print(f"time taken: {time_alg}\tmhr: {mhrs[7]}\n")

    # HS directly
    selected, time_alg = HSAlgorithm.run_hs(dataP=data_points, r=k, k=1, curSky=[])
    result_hs = Algorithms.find_point_indices(selected, data_points)
    mhr = calculate_mhr(data_points, result_hs, [])
    mhrs[12] = mhr
    times[12] = time_alg
    print(f"time taken: {time_alg}\tmhr: {mhrs[12]}\n")

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

def run_eld_experiments(data_points: List[Point], grouped_data: Dict[int, List[Point]],
                       fairness_constraints: Dict[int, Any], group_id: int,
                       utility_funcs: List[List[float]], dataset: str) -> None:
    # Varying Epsilon and Lambda
    deltaC = 0.25
    setDelta = 1
    fairness_constraints = generate_fairness_constraints(grouped_data, data_points, 8)
    mhrs_el, times_el = [], []

    for epsilon in [0.000625 * (2**i) for i in range(10) if 0.000625 * (2**i) < 0.7]:
        mhr_row, time_row = [], []
        for lambda_val in [0.000625 * (2**i) for i in range(10) if 0.000625 * (2**i) < 0.7]:
            result, time_alg = BiGreedy.run_bi_greedy_plus_with_delta(
                grouped_data, fairness_constraints, data_points, group_id, lambda_val, epsilon,
                utility_funcs, 8, deltaC, setDelta)
            mhr = calculate_mhr(data_points, result, utility_funcs)
            mhr_row.append(mhr)
            time_row.append(time_alg)
        mhrs_el.append(mhr_row)
        times_el.append(time_row)

    out_file = f"../result/{dataset[:-4]}_8_DC{deltaC:.2f}.data"
    with open(out_file, 'a') as fout:
        for row in mhrs_el:
            fout.write("\t".join(f"{mhr:8.2f}" for mhr in row) + "\n")
        for row in times_el:
            fout.write("\t".join(f"{t:8.2f}" for t in row) + "\n")

    # Varying Delta
    lambda_val = 0.04
    setDelta = 0
    fairness_constraints = generate_fairness_constraints(grouped_data, data_points, 15)
    mhrs_d, times_d = [], []
    pmhrs, ptimes = [], []

    for delta in [1.25 * (2**i) for i in range(6) if 1.25 * (2**i) < 60]:
        # Bi-Greedy with Delta
        result, time_alg = BiGreedy.run_bi_greedy_with_delta(
            grouped_data, fairness_constraints, data_points, group_id, 0.02, utility_funcs, 15, delta*10)
        mhr = calculate_mhr(data_points, result, utility_funcs)
        mhrs_d.append(mhr)
        times_d.append(time_alg)

        # Bi-Greedy-Plus with Delta
        result, time_alg = BiGreedy.run_bi_greedy_plus_with_delta(
            grouped_data, fairness_constraints, data_points, group_id, lambda_val, 0.02,
            utility_funcs, 15, delta)
        pmhr = calculate_mhr(data_points, result, utility_funcs)
        pmhrs.append(pmhr)
        ptimes.append(time_alg)

    out_file = f"../result/{dataset[:-4]}_15_E0.02_L0.04.data"
    with open(out_file, 'a') as fout:
        fout.write("\t".join(f"{mhr:8.2f}" for mhr in mhrs_d) + "\n")
        fout.write("\t".join(f"{mhr:8.2f}" for mhr in pmhrs) + "\n")
        fout.write("\t".join(f"{t:8.2f}" for t in times_d) + "\n")
        fout.write("\t".join(f"{t:8.2f}" for t in ptimes) + "\n")

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

        # run_baselines(data_points, grouped_data, fairness_constraints, group_id, args.k, mhrs, times)
        write_aggregated_results(args.dataset, args.k, mhrs, times)

        # if args.k != 2 :
        #     run_eld_experiments(data_points, grouped_data, fairness_constraints, group_id, utility_funcs, args.dataset)

if __name__ == "__main__":
    main()