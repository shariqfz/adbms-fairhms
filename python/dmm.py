import math
import numpy as np
import time
from typing import List, Tuple
from heapq import nlargest
from bisect import bisect_left
from point import Point

class DMM:
    @staticmethod
    def discretize(gamma: int, dim: int) -> List[Point]:
        size = int(gamma ** (dim - 1))
        alpha = math.pi / (2 * gamma)
        points = []
        theta = [1] * (dim - 1)
        
        for _ in range(size):
            coords = np.zeros(dim)
            r = 1.0
            for j in reversed(range(1, dim)):
                angle = (theta[j-1] - 0.5) * alpha
                coords[j] = r * math.cos(angle)
                r *= math.sin(angle)
            coords[0] = r
            points.append(Point(dim, coord=coords.tolist()))
            
            # Update theta
            for j in range(dim-1):
                if theta[j] < gamma:
                    theta[j] += 1
                    break
                else:
                    theta[j] = 1
        return points

    @staticmethod
    def mrst_oracle(M: List[List[float]], points: List[Point], eps: float) -> List[Point]:
        n = len(points)
        m = len(M[0]) if n > 0 else 0
        covered = [False] * m
        selected = []
        
        # Create coverage matrix
        coverage = [ [i for i, val in enumerate(row) if val <= eps] for row in M ]
        remaining = set(range(m))
        
        while remaining:
            best = -1
            best_count = 0
            for i in range(n):
                if points[i] in selected:
                    continue
                cnt = len([j for j in coverage[i] if j in remaining])
                if cnt > best_count:
                    best = i
                    best_count = cnt
            if best == -1:
                break
            selected.append(points[best])
            remaining -= set(coverage[best])
        return selected

    @staticmethod
    def dmm(points: List[Point], k: int) -> List[Point]:
        if not points:
            return []
        dim = points[0].dim
        gamma = 5
        F = DMM.discretize(gamma, dim)
        
        n = len(points)
        m = len(F)
        M = [[0.0]*m for _ in range(n)]
        
        # Compute M matrix
        for j in range(m):
            max_dot = max(p.dot_product(F[j]) for p in points)
            for i in range(n):
                M[i][j] = 1 - points[i].dot_product(F[j]) / max_dot if max_dot != 0 else 0
                
        # Get sorted unique errors
        errors = sorted({M[i][j] for i in range(n) for j in range(m)})
        low, high = 0, len(errors)
        best_solution = []
        
        # Binary search
        while low < high:
            mid = (low + high) // 2
            eps = errors[mid]
            solution = DMM.mrst_oracle(M, points, eps)
            if len(solution) <= k:
                best_solution = solution
                high = mid
            else:
                low = mid + 1
        return best_solution[:k]

    @staticmethod
    def dmm_greedy(points: List[Point], k: int) -> List[Point]:
        if not points:
            return []
        dim = points[0].dim
        if k < dim:
            raise ValueError("k must be >= dimension")
        
        # Select extreme points
        selected = []
        for d in range(dim):
            max_val = -np.inf
            best_p = None
            for p in points:
                if p.coord[d] > max_val:
                    max_val = p.coord[d]
                    best_p = p
            if best_p and best_p not in selected:
                selected.append(best_p)
        
        # Greedily add remaining points
        remaining = [p for p in points if p not in selected]
        gamma = 10
        F = DMM.discretize(gamma, dim)
        m = len(F)
        
        # Precompute M matrix
        M = [[0.0]*m for _ in range(len(points))]
        for j in range(m):
            max_dot = max(p.dot_product(F[j]) for p in points)
            for i, p in enumerate(points):
                M[i][j] = 1 - p.dot_product(F[j]) / max_dot if max_dot != 0 else 0
                
        while len(selected) < k and remaining:
            worst_dir = None
            min_coverage = np.inf
            # Find worst covered direction
            for j in range(m):
                current_min = min(M[i][j] for i, p in enumerate(points) if p in selected)
                if current_min < min_coverage:
                    min_coverage = current_min
                    worst_dir = j
            
            # Find best point to improve worst direction
            best_point = None
            best_value = np.inf
            for p in remaining:
                val = M[points.index(p)][worst_dir]
                if val < best_value:
                    best_value = val
                    best_point = p
            if best_point:
                selected.append(best_point)
                remaining.remove(best_point)
        return selected[:k]

    @staticmethod
    def run_dmm_greedy(dataP: List[Point], r: int, k: int) -> Tuple[List[Point], float]:
        start = time.time()
        result = DMM.dmm_greedy(dataP, r)
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        return result, elapsed

    @staticmethod
    def run_dmm_rrms(dataP: List[Point], r: int, k: int) -> Tuple[List[Point], float]:
        start = time.time()
        result = DMM.dmm(dataP, r)
        elapsed = (time.time() - start) * 1000
        return result, elapsed