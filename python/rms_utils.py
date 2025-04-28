import math
import numpy as np
import heapq
from typing import List, Tuple
from point import Point

class RMSUtils:
    # Static variables
    dimension = 1
    transformation_matrix = np.eye(1)
    center = np.zeros(1)
    
    _ref_to_pts = None
    _ref_to_dir = None

    @staticmethod
    def log_net_size(sphere_radius: float, net_radius: float, dim: int) -> int:
        assert sphere_radius > 0 and net_radius > 0
        return 1 + (dim - 1) * (2 + math.ceil(math.log2(sphere_radius / net_radius)))

    @staticmethod
    def log_random_net_size(sphere_radius: float, net_radius: float, delta: float, dim: int) -> int:
        assert dim > 0 and sphere_radius > 0 and net_radius > 0 and 0 < delta < 1
        logM = RMSUtils.log_net_size(sphere_radius, net_radius/2, dim)
        val = logM + math.log2(logM - math.log(delta))
        return math.ceil(val)

    @staticmethod
    def get_random_sphere_points(sphere_radius: float, dim: int, N: int, 
                                first_orthant: bool) -> List[Point]:
        points = []
        for _ in range(N):
            vec = np.random.normal(size=dim)
            if first_orthant:
                vec = np.abs(vec)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm * sphere_radius
            points.append(Point(dim, coordinates=vec.tolist()))
        return points

    @staticmethod
    def max_avg_regret(dataP: List[Point], R: List[Point], k: int, N: int = 10000) -> Tuple[float, float, float]:
        dim = dataP[0].dim if dataP else 1
        random_dirs = RMSUtils.get_random_sphere_points(1.0, dim, N, True)
        scores = []
        max_regret = 0.0
        total_regret = 0.0

        for dir in random_dirs:
            maxQ = max(p.dot_product(dir) for p in R) if R else 0
            topk = heapq.nlargest(k, (p.dot_product(dir) for p in dataP))
            maxPk = topk[-1] if topk else 0
            
            if maxPk > maxQ:
                delta = maxQ / maxPk
                regret = 1 - delta
                scores.append(regret)
                total_regret += regret
                max_regret = max(max_regret, regret)
            else:
                scores.append(0)

        scores.sort()
        perc80 = scores[int(0.8 * len(scores))] if scores else 0
        return max_regret, total_regret/N, perc80

    @staticmethod
    def ndir_for_validation(dim: int) -> int:
        if dim == 1: return 2
        elif dim <= 3: return 20000
        elif dim <= 4: return 60000
        elif dim <= 6: return 80000
        elif dim <= 8: return 120000
        elif dim <= 12: return 200000
        elif dim <= 16: return 200000
        elif dim <= 20: return 262144
        else: return 270500

    @staticmethod
    def rank_selection_dotp(pts: List[Point], dir: Point, k: int) -> Tuple[List[int], List[float]]:
        if k <= math.log2(len(pts)):
            return RMSUtils._naive_rank_selection(pts, dir, k)
        return RMSUtils._heap_rank_selection(pts, dir, k)

    @staticmethod
    def _naive_rank_selection(pts: List[Point], dir: Point, k: int) -> Tuple[List[int], List[float]]:
        scored = [(p.dot_product(dir), i) for i, p in enumerate(pts)]
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]], [s for s, _ in scored[:k]]

    @staticmethod
    def _heap_rank_selection(pts: List[Point], dir: Point, k: int) -> Tuple[List[int], List[float]]:
        heap = [(-p.dot_product(dir), i) for i, p in enumerate(pts)]
        heapq.heapify(heap)
        topk = [heapq.heappop(heap) for _ in range(k)]
        return [i for _, i in topk], [-s for s, _ in topk]

    @staticmethod
    def stavros_transform(dataP: List[Point]) -> List[Point]:
        if not dataP:
            return []
        dim = dataP[0].dim
        max_coords = [max(p.coord[i] for p in dataP) for i in range(dim)]
        return [Point(dim, coord=[p.coord[i]/max_coords[i] for i in range(dim)]) for p in dataP]

    @staticmethod
    def get_fat_pointset2(dataP: List[Point]) -> Tuple[List[Point], float, float]:
        fatP = RMSUtils.stavros_transform(dataP)
        dim = dataP[0].dim if dataP else 1
        c = dim + math.sqrt(dim)
        center = Point(dim, coord=[1.0/c]*dim)
        fatP = [p - center for p in fatP]
        return fatP, 1.0, 1.0/c

    @staticmethod
    def point_set_transf(fatP: List[Point]) -> List[Point]:
        return [Point(p.dim, p.id, p.coord) for p in fatP]