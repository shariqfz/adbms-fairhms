import os
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

@dataclass
class FairGroup:
    lc: int
    uc: int
    ki: int

def split(src: str, delimiter: str) -> List[str]:
    parts = []
    tmp = []
    i = 0
    while i < len(src):
        if src[i] != delimiter:
            tmp.append(src[i])
            i += 1
        else:
            parts.append(''.join(tmp))
            tmp = []
            i += 1
    if tmp:
        parts.append(''.join(tmp))
    return parts

def read_data_points(file_name: str) -> Tuple[List[Point], int, int, List[Dict[str, int]]]:
    data_p = []
    dim = 0
    group_dim = 0
    group_to_int = []
    with open(file_name, 'r') as fin:
        lines = fin.read().splitlines()
        if not lines:
            return data_p, dim, group_dim, group_to_int
        dim = int(lines[0])
        group_dim = 0
        group_ids = []
        for line in lines[1:]:
            parts = split(line, '\t')
            current_group_dim = len(parts) - dim
            if group_dim == 0:
                group_dim = current_group_dim
                group_to_int = [{} for _ in range(group_dim)]
                group_ids = [0] * group_dim
            coords = list(map(float, parts[:dim]))
            groups = []
            for j in range(group_dim):
                category = parts[dim + j]
                if category not in group_to_int[j]:
                    group_to_int[j][category] = group_ids[j]
                    group_ids[j] += 1
                groups.append(group_to_int[j][category])
            data_p.append(Point(dim, len(data_p), coords, groups))
    return data_p, dim, group_dim, group_to_int

def read_utility_functions(file_name: str, m: int) -> List[Point]:
    utilities = []
    with open(file_name, 'r') as fin:
        for count, line in enumerate(fin):
            if count >= m:
                break
            coords = list(map(float, line.strip().split()))
            utilities.append(Point(len(coords), count, coords))
    return utilities

def generate_groups(data_p: List[Point], group_id: int) -> Dict[int, List[Point]]:
    grouped = defaultdict(list)
    for p in data_p:
        cat = p.get_category(group_id)
        grouped[cat].append(p)
    return grouped

def generate_fair_constraints(grouped_data: Dict[int, List[Point]], data_p: List[Point], k: int) -> Dict[int, FairGroup]:
    fairness = defaultdict(lambda: FairGroup(0, 0, 0))
    total = len(data_p)
    
    # Calculate proportional ki values
    proportions = {gid: len(points)/total for gid, points in grouped_data.items()}
    for gid in grouped_data:
        fairness[gid].ki = round(k * proportions[gid])
    
    # Adjust ki to sum to k
    total_ki = sum(fc.ki for fc in fairness.values())
    while total_ki != k:
        if total_ki < k:
            gid = min(fairness, key=lambda x: fairness[x].ki)
            fairness[gid].ki += 1
            total_ki += 1
        else:
            gid = max(fairness, key=lambda x: fairness[x].ki)
            if fairness[gid].ki > 1:
                fairness[gid].ki -= 1
                total_ki -= 1
            else:
                break

    # Set lower and upper bounds
    gap = max(1, int(0.1 * k))
    for gid in fairness:
        fairness[gid].lc = max(1, fairness[gid].ki - gap)
        fairness[gid].uc = fairness[gid].ki + gap
        
    return fairness

def write_to_file(data_p: List[Point], dim: int, group_to_int: List[Dict[str, int]],
                 fairness_constraint: Dict[int, FairGroup], dataset_path: str,
                 result: List[int], k: int, group_id: int, alg_name: str,
                 time: float, mhr: float):
    os.makedirs('../result', exist_ok=True)
    base_name = os.path.basename(dataset_path).rsplit('.', 1)[0]
    out_path = f"../result/{base_name}_{k}.txt"
    
    with open(out_path, 'a') as fout:
        # Write header
        fout.write(f"{alg_name} groupID={group_id}\nFairness constraint: ")
        
        # Map group IDs to names
        group_names = {}
        for name, code in group_to_int[group_id].items():
            group_names[code] = name
            
        # Write fairness constraints
        for gid in sorted(fairness_constraint.keys()):
            fc = fairness_constraint[gid]
            fout.write(f"{group_names[gid]}={fc.lc}-{fc.ki}-{fc.uc} ")
        fout.write("\n")
        
        # Write selected points
        counts = defaultdict(int)
        for p_idx in result:
            p = data_p[p_idx]
            cat = p.get_category(group_id)
            counts[cat] += 1
            fout.write(f"idx={p_idx}\t{group_names[cat]}\t")
            fout.write("\t".join(f"{p.get_coordinate(i):.6f}" for i in range(dim)))
            fout.write("\n")
        
        # Write counts
        for gid in sorted(fairness_constraint.keys()):
            fout.write(f"{group_names[gid]}={counts[gid]} ")
        fout.write(f"\nHR={mhr:.4f}\nTime={time:.2f}\n\n")

# Similar implementation for write_to_file_eld would follow the same pattern
# with additional parameters handled in the output filename and headers