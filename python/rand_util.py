import numpy as np
from typing import Optional
from .Point import Point  # Assuming Point class is defined as before

class RandUtil:
    _iset: Optional[bool] = None
    _gset: float = 0.0

    @staticmethod
    def get_random_direction(dim: int) -> Point:
        """Generate a random unit vector using Box-Muller transform"""
        while True:
            coords = []
            # Generate using Box-Muller to match original C++ implementation
            if not RandUtil._iset:
                r = 2.0
                while r >= 1.0:
                    v1 = np.random.uniform(-1, 1)
                    v2 = np.random.uniform(-1, 1)
                    r = v1**2 + v2**2
                
                fac = np.sqrt(-2 * np.log(r) / r)
                RandUtil._gset = v1 * fac
                RandUtil._iset = True
                coords.append(v2 * fac)
            else:
                coords.append(RandUtil._gset)
                RandUtil._iset = False
            
            # Fill remaining coordinates with normal distribution
            while len(coords) < dim:
                coords.append(np.random.normal())
            
            arr = np.array(coords)
            norm = np.linalg.norm(arr)
            if norm > 1e-8:  # Avoid division by near-zero
                return Point(dim, coord=(arr / norm).tolist())

    @staticmethod
    def rand_unif(lo: float, hi: float) -> float:
        """Generate uniform random number in [lo, hi)"""
        return np.random.uniform(lo, hi)