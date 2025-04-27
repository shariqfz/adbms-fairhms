import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Set
import time
from rms_utils import RMSUtils  
from point import Point  

class EpsKernel:
    @staticmethod
    def validate_coreset(fatP: List[Point], idxs: List[int], epsilon: float, dim: int, k: int) -> bool:
        if not fatP:
            return True
        if not idxs:
            return False

        ndir = RMSUtils.ndir_for_validation(dim)
        random_dirs = RMSUtils.get_random_sphere_points(1.0, dim, ndir, True)

        for dir in random_dirs:
            core_max = max(fatP[i].dot_product(dir) for i in idxs)
            count = 0
            for p in fatP:
                val = p.dot_product(dir)
                if core_max < (1 - epsilon) * val:
                    count += 1
                    if count >= k:
                        return False
        return True

    @staticmethod
    def coreset_by_sample(ann: NearestNeighbors, fatP: List[Point], outer_rad: float, 
                          delta: float, sample_size: int, k: int) -> Set[int]:
        dim = fatP[0].dim
        random_points = RMSUtils.get_random_sphere_points(outer_rad, dim, sample_size, True)
        
        # Convert Points to numpy array for sklearn
        data = np.array([p.coord for p in fatP])
        ann.fit(data)
        
        # Find k nearest neighbors for each random point
        _, indices = ann.kneighbors(np.array([p.coord for p in random_points]), n_neighbors=k)
        
        unique_ids = set()
        for row in indices:
            for idx in row:
                unique_ids.add(idx)
        return unique_ids

    @staticmethod
    def get_coreset(fatP: List[Point], epsilon: float, k: int, skyline: List[Point]) -> Tuple[List[Point], float]:
        if not fatP:
            return [], 0.0

        dim = fatP[0].dim
        if len(fatP) < dim:
            return fatP, 0.0

        outer_rad = 1 + np.sqrt(dim)
        delta = epsilon / (2 * outer_rad)
        ann = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        
        unique_ids = set()
        sample_size = 10
        coreset = []
        start_time = time.time()

        while True:
            # Generate new candidates
            new_ids = EpsKernel.coreset_by_sample(ann, fatP, outer_rad, delta, sample_size, k)
            unique_ids.update(new_ids)
            
            # Convert to sorted index list
            idxs = sorted(unique_ids)
            coreset = [fatP[i] for i in idxs]
            
            # Validate coreset
            if EpsKernel.validate_coreset(fatP, idxs, epsilon, dim, k):
                break
                
            sample_size *= 2

        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        return coreset, elapsed

    @staticmethod
    def run_eps_kernel(dataP: List[Point], r: int, k: int, curSky: List[Point]) -> Tuple[List[Point], float]:
        print("Running Eps-Kernel...")
        dim = dataP[0].dim if dataP else 0
        result = []
        time_used = 0.0
        
        # Binary search for optimal epsilon
        lower, upper = 0.0, 1.0
        epsilon = (upper + lower) / 2
        max_iterations = 20
        
        for _ in range(max_iterations):
            coreset, t = EpsKernel.get_coreset(dataP, epsilon, k, curSky)
            core_size = len(coreset)
            
            if core_size <= r:
                upper = epsilon
            else:
                lower = epsilon
                
            epsilon = (upper + lower) / 2
            if upper - lower < 0.001:
                break
        
        # Get final coreset with best epsilon
        final_coreset, final_time = EpsKernel.get_coreset(dataP, epsilon, k, curSky)
        return final_coreset[:r], final_time