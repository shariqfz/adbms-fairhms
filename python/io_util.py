import os
from typing import List, Tuple
from .Point import Point

class IOUtil:
    @staticmethod
    def read_input_points(fname: str) -> Tuple[int, List[Point]]:
        """Read points from file, first line is dimension"""
        points = []
        dim = 0
        try:
            with open(fname, 'r') as f:
                # Read dimension from first line
                dim = int(f.readline().strip())
                count = 0
                for line in f:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) != dim:
                        raise ValueError(f"Invalid coordinate count in line {count+1}")
                    points.append(Point(dim, count, coords))
                    count += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot open file {fname} for reading")
        return dim, points

    @staticmethod
    def write_output_points(fname: str, dim: int, dataP: List[Point]):
        """Write points to file with dimension header"""
        try:
            with open(fname, 'w') as f:
                f.write(f"{dim}\n")
                for i, p in enumerate(dataP):
                    if p.dim != dim:
                        raise ValueError(f"Point {i} has wrong dimension")
                    line = " ".join(f"{c:.6f}" for c in p.coord)
                    f.write(line + ("\n" if i < len(dataP)-1 else ""))
        except IOError:
            raise IOError(f"Cannot open file {fname} for writing")

    @staticmethod
    def read_utils(fname: str, dim: int, size: int) -> List[Point]:
        """Read specified number of utility points"""
        utils = []
        try:
            with open(fname, 'r') as f:
                for _ in range(size):
                    coords = []
                    while len(coords) < dim:
                        line = f.readline()
                        if not line:
                            raise ValueError("Unexpected end of file")
                        coords.extend(map(float, line.strip().split()))
                    utils.append(Point(dim, len(utils), coords[:dim]))
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot open file {fname} for reading")
        return utils

    @staticmethod
    def read_all_utils(fname: str, dim: int) -> List[Point]:
        """Read all utility points from file"""
        return IOUtil.read_utils(fname, dim, float('inf'))

    @staticmethod
    def read_random_utils(fname: str, dim: int, size: int, random_num: int) -> List[Point]:
        """Read utilities with random offset"""
        utils = []
        try:
            with open(fname, 'r') as f:
                # Skip random_num lines
                for _ in range(random_num):
                    f.readline()
                # Read specified number of utilities
                utils = IOUtil.read_utils(fname, dim, size)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot open file {fname} for reading")
        return utils

    @staticmethod
    def read_fixed_points(fname: str) -> Tuple[int, List[Point], List[int], List[bool]]:
        """Read points with insertion tracking"""
        dim, all_points = IOUtil.read_input_points(fname)
        size = len(all_points) // 2
        dataP = all_points[:size]
        to_insert = [p.id for p in all_points[size:]]
        is_deleted = [False]*size + [True]*(len(all_points)-size)
        return dim, dataP, to_insert, is_deleted