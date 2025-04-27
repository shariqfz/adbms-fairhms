import os
import math
from typing import List, Dict, Tuple
from collections import defaultdict, namedtuple
import numpy as np
from point import Point
from lp import evaluateLP, PointSet
from rms_utils import RMSUtils

FairGroup = namedtuple('FairGroup', ['lc', 'uc', 'ki'])

def split_string(src: str, delimiter: str) -> List[str]:
    return src.split(delimiter)

def read_data_points(file_path: str) -> Tuple[List[Point], int, int, List[Dict[str, int]]]:
    data_points = []
    group_mappings = []
    dim = 0
    group_dim = 0
    
    with open(file_path, 'r') as f:
        # Read dimensionality from first line
        dim = int(f.readline().strip())
        group_mappings = [defaultdict(int) for _ in range(5)]  # Initialize for max 5 groups
        group_ids = [0] * 5
        
        for count, line in enumerate(f):
            parts = line.strip().split('\t')
            coords = list(map(float, parts[:dim]))
            
            # Handle group dimensions
            group_dim = len(parts) - dim
            groups = []
            for j in range(group_dim):
                category = parts[dim + j]
                if category not in group_mappings[j]:
                    group_mappings[j][category] = group_ids[j]
                    group_ids[j] += 1
                groups.append(group_mappings[j][category])
            
            data_points.append(Point(dim, count, coords, groups))
    
    return data_points, dim, group_dim, group_mappings

def read_utility_functions(file_path: str, m: int) -> List[Point]:
    utilities = []
    with open(file_path, 'r') as f:
        for count, line in enumerate(f):
            if count >= m:
                break
            coords = list(map(float, line.strip().split()))
            utilities.append(Point(len(coords), count, coords))
    return utilities

def generate_groups(data: List[Point], group_id: int) -> Dict[int, List[Point]]:
    groups = defaultdict(list)
    for point in data:
        group_key = point.get_category(group_id)
        groups[group_key].append(point)
    return groups

def write_results(data: List[Point], dim: int, group_mappings: List[Dict[str, int]], 
                 fairness_constraints: Dict[int, FairGroup], dataset_path: str, 
                 result_indices: List[int], k: int, group_id: int, algorithm: str, 
                 time_taken: float, mhr: float):
    # Create output directory if needed
    os.makedirs('../result', exist_ok=True)
    
    # Prepare output file path
    base_name = os.path.basename(dataset_path).rsplit('.', 1)[0]
    output_path = f"../result/{base_name}_{k}.txt"
    
    # Convert points to PointSet format for HR calculation
    fat_point_set = PointSet(len(data), dim)
    result_point_set = PointSet(len(result_indices), dim)
    
    for i, idx in enumerate(result_indices):
        result_point_set.points[i] = data[idx]
    
    with open(output_path, 'a') as f:
        f.write(f"{algorithm} groupID={group_id}\n")
        f.write("Fairness constraints: ")
        
        # Map group IDs to names
        group_names = {}
        for group_id, mapping in enumerate(group_mappings):
            group_names[group_id] = {v: k for k, v in mapping.items()}
        
        # Write fairness constraints
        for gid, constraint in fairness_constraints.items():
            f.write(f"{group_names[group_id][gid]}={constraint.lc}-{constraint.ki}-{constraint.uc} ")
        f.write("\n")
        
        # Write selected points
        category_counts = defaultdict(int)
        for idx in result_indices:
            point = data[idx]
            category = point.get_category(group_id)
            category_name = group_names[group_id][category]
            f.write(f"idx={idx}\t{category_name}\t")
            f.write("\t".join(map(str, point.coordinates)) )
            f.write("\n")
            category_counts[category] += 1
        
        # Write category counts
        for gid, count in category_counts.items():
            f.write(f"{group_names[group_id][gid]}={count} ")
        f.write("\n")
        
        f.write(f"HR={mhr}\nTime={time_taken}\n\n")

# def generate_fairness_constraints(groups: Dict[int, List[Point]], 
#                                  data: List[Point], k: int) -> Dict[int, FairGroup]:
#     constraints = {}
#     total_points = len(data)
#     alpha = 0.1
#     gap = max(1, int(alpha * k))
    
#     # Calculate initial proportions
#     proportions = {}
#     for gid, groups in groups.items():
#         proportions[gid] = len(groups) / total_points
    
#     # Initial ki values
#     total = 0
#     for gid in groups:
#         ki = round(k * proportions[gid])
#         constraints[gid] = FairGroup(lc=0, uc=0, ki=ki)
#         total += ki
    
#     # Adjust ki values to sum to k
#     while total != k:
#         diff = k - total
#         step = 1 if diff > 0 else -1
#         for gid in sorted(groups.keys(), key=lambda x: constraints[x].ki):
#             if diff == 0:
#                 break
#             constraints[gid] = constraints[gid]._replace(ki=constraints[gid].ki + step)
#             total += step
#             diff -= step
    
#     # Set lower and upper bounds
#     for gid in constraints:
#         lc = max(1, constraints[gid].ki - gap)
#         uc = constraints[gid].ki + gap
#         constraints[gid] = constraints[gid]._replace(lc=lc, uc=uc)
    
#     return constraints

def generate_fairness_constraints(groups: Dict[int, List[Point]], 
                                 data: List[Point], k: int) -> Dict[int, FairGroup]:
    constraints = {}
    total_points = len(data)
    alpha = 0.1
    gap = max(1, int(alpha * k))
    
    # Get sorted group IDs (C++ uses consecutive indexes)
    group_ids = sorted(groups.keys())
    
    # Calculate proportions using sorted groups
    proportions = {gid: len(groups[gid]) / total_points for gid in group_ids}
    
    # Initial ki values
    total = 0
    for gid in group_ids:
        ki = round(k * proportions[gid])
        constraints[gid] = FairGroup(lc=0, uc=0, ki=ki)
        total += ki
    
    # Adjustment logic needs to match C++'s vector index approach
    while total != k:
        diff = k - total
        step = 1 if diff > 0 else -1
        
        # Sort groups by current ki (like C++ does)
        sorted_groups = sorted(group_ids, key=lambda x: constraints[x].ki)
        
        for gid in sorted_groups:
            if diff == 0:
                break
            new_ki = constraints[gid].ki + step
            if new_ki >= 0:  # Prevent negative ki
                constraints[gid] = constraints[gid]._replace(ki=new_ki)
                total += step
                diff -= step
    
    # Set bounds using C++ logic
    for gid in group_ids:
        lc = max(1, constraints[gid].ki - gap)
        uc = constraints[gid].ki + gap
        constraints[gid] = constraints[gid]._replace(lc=lc, uc=uc)
    
    return constraints

# ELD version with additional parameters
def write_results_eld(data: List[Point], dim: int, group_mappings: List[Dict[str, int]], 
                     fairness_constraints: Dict[int, FairGroup], dataset_path: str, 
                     result_indices: List[int], k: int, group_id: int, algorithm: str, 
                     time_taken: float, mhr: float, epsilon: float, lambda_val: float, 
                     delta_c: float, set_delta: int):
    output_path = f"../result/{os.path.basename(dataset_path).rsplit('.', 1)[0]}_"
    if set_delta == 1:
        output_path += f"{k}_DC{delta_c:.2f}.txt"
    else:
        output_path += f"{k}_E{epsilon:.2f}_L{lambda_val:.2f}.txt"
    
    with open(output_path, 'a') as f:
        header = f"{algorithm} groupID={group_id}"
        if set_delta == 1:
            header += f"\tDeltaC={delta_c:.2f}"
        else:
            header += f"\tEpsilon={epsilon:.2f}\tLambda={lambda_val:.2f}"
        f.write(header + "\n")
        
        # Rest of the implementation similar to write_results...
        # ... (omitted for brevity, would include same core logic as write_results)

def get_skyline(data: List[Point]) -> List[Point]:
    skyline = []
    for point in data:
        dominated = False
        to_remove = []
        for i, sk_point in enumerate(skyline):
            if sk_point.dominates(point):
                dominated = True
                break
            if point.dominates(sk_point):
                to_remove.append(i)
        if not dominated:
            skyline = [p for i, p in enumerate(skyline) if i not in to_remove]
            skyline.append(point)
    return skyline