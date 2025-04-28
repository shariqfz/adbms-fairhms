import numpy as np
from typing import List, Set, Dict, Tuple
from point import Point
from rms_utils import RMSUtils
import time 
from tqdm import tqdm

class HSApprox:
    @staticmethod
    def get_hs_approximation(S: List[Set[int]]) -> List[int]:
        """
        Greedy approximation algorithm for hitting set problem
        """
        element_coverage: Dict[int, Set[int]] = {}
        
        # Build element coverage map
        for set_id, current_set in enumerate(S):
            for element in current_set:
                if element not in element_coverage:
                    element_coverage[element] = set()
                element_coverage[element].add(set_id)
        
        uncovered = set(range(len(S)))
        hitting_set = []
        
        while uncovered:
            best_element = -1
            best_coverage = set()
            max_covered = 0
            
            # Find element covering most uncovered sets
            for element, covered_sets in element_coverage.items():
                current_covered = covered_sets & uncovered
                if len(current_covered) > max_covered:
                    max_covered = len(current_covered)
                    best_element = element
                    best_coverage = current_covered
            
            if best_element == -1:
                break  # No covering element found
            
            # Update uncovered sets and add to hitting set
            uncovered -= best_coverage
            hitting_set.append(best_element)
            del element_coverage[best_element]
        
        return hitting_set

class HSAlgorithm:
    @staticmethod
    def validate_hs(fatP: List[Point], idxs: List[int], k: int, 
                   epsilon: float, dim: int) -> bool:
        """
        Validate if hitting set meets epsilon-regret condition
        """
        if not fatP or not idxs:
            return False
        
        ndir = RMSUtils.ndir_for_validation(dim)
        random_dirs = RMSUtils.get_random_sphere_points(1.0, dim, ndir, True)
        
        for direction in tqdm(random_dirs, desc="Validation directions"):
            core_max = max(fatP[i].dot_product(direction) for i in idxs)
            count = 0
            
            for p in fatP:
                pt_val = p.dot_product(direction)
                if core_max < (1 - epsilon) * pt_val:
                    count += 1
                    if count >= k:
                        return False
        return True

    @staticmethod
    def hs_by_sampling(fatP: List[Point], dim: int, k: int, 
                      epsilon: float, sample_size: int, idxs: List[int]) -> List[int]:
        """
        Generate candidate sets and find hitting set
        """
        # Generate random directions
        random_dirs = RMSUtils.get_random_sphere_points(1.0, dim, sample_size, True)
        candidate_sets = []
        
        for direction in random_dirs:
            # Get top 10k points in this direction
            k1 = min(10 * k, len(fatP))
            topk_indices = RMSUtils.rank_selection_dotp(fatP, direction, k1)[0]
            topk_values = [fatP[i].dot_product(direction) for i in topk_indices]
            
            if not topk_values:
                continue  # Handle empty case
            
            # Threshold for inclusion
            threshold = (1 - epsilon) * topk_values[k-1]
            
            # Check if any existing idxs cover this direction
            existing_cover = False
            for idx in idxs:
                if fatP[idx].dot_product(direction) >= threshold:
                    existing_cover = True
                    break
            if existing_cover:
                continue  # Skip adding this candidate set
            
            candidate_set = set()
            for i, val in zip(topk_indices, topk_values):
                if val >= threshold:
                    candidate_set.add(i)
                else:
                    break
            
            if candidate_set:
                candidate_sets.append(candidate_set)
        
        # Get hitting set approximation
        return HSApprox.get_hs_approximation(candidate_sets)


    @staticmethod
    def run_hs(dataP: List[Point], r: int, k: int, 
              curSky: List[Point]) -> Tuple[List[Point], float]:
        """
        Main HS algorithm with epsilon binary search
        """
        print("Running HS...")
        start_time = time.time()
        result = []
        
        if not dataP or r < 1:
            return [], 0.0
        
        dim = dataP[0].dimension
        epsilon = 0.5
        lower, upper = 0.0, 1.0
        
        for _ in tqdm(range(20), desc="Binary search"):  # Max binary search iterations
            sample_size = 10  # Reset sample_size each iteration
            idxs = []
            hsP = []
            current_size = 0
            
            while True:
                # Get new candidates with current idxs
                new_idxs = HSAlgorithm.hs_by_sampling(dataP, dim, k, epsilon/5, sample_size, idxs)
                idxs = list(set(idxs + new_idxs))
                hsP = [dataP[i] for i in idxs]
                
                # Validate solution
                if k > 1:
                    valid = HSAlgorithm.validate_hs(dataP, idxs, k, epsilon, dim)
                else:
                    valid = HSAlgorithm.validate_hs(curSky, idxs, k, epsilon, dim)
                
                if valid:
                    current_size = len(idxs)
                    break
                
                sample_size *= 2
            
            # Binary search adjustment
            if current_size > r + 5:
                lower = epsilon
                epsilon = (epsilon + upper) / 2
            else:
                upper = epsilon
                epsilon = (epsilon + lower) / 2
            
            if upper - lower < 0.001:
                break
        
        # Final result
        final_result = hsP[:r]
        elapsed = (time.time() - start_time) * 1000  # ms
        return final_result, elapsed