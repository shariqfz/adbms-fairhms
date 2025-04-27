class Point:
    def __init__(self, dim: int, id: int = -1, coord: list[float] = None):
        self.dim = dim
        self.id = id
        self.coord = coord if coord is not None else [0.0] * dim
        
    def __repr__(self) -> str:
        return f"Point(id={self.id}, dim={self.dim}, coord={self.coord})"
    
    def print_point(self) -> None:
        """Print point coordinates in C++ style format"""
        print(f"{self.id} ", end="")
        print(' '.join(f"{c:.6f}" for c in self.coord))


class PointSet:
    def __init__(self, points: list[Point] = None):
        self.points = points if points is not None else []
        self.number_of_points = len(self.points)
        
    def __repr__(self) -> str:
        return f"PointSet(n={self.number_of_points})"
    
    def print_point_set(self) -> None:
        """Print point IDs in C++ style format"""
        print(' '.join(str(p.id) for p in self.points))
        print()

    def add_point(self, point: Point) -> None:
        self.points.append(point)
        self.number_of_points += 1


# Utility functions matching C++ interface style
def alloc_point(dim: int, id: int = -1) -> Point:
    """Create a new Point with specified dimension"""
    return Point(dim=dim, id=id)

def alloc_point_set() -> PointSet:
    """Create an empty PointSet"""
    return PointSet()

def print_point(point: Point) -> None:
    """Standalone print function matching C++ interface"""
    point.print_point()

def print_point_set(point_set: PointSet) -> None:
    """Standalone print function matching C++ interface"""
    point_set.print_point_set()


# Example usage
if __name__ == "__main__":
    # Create points
    p1 = Point(3, 0, [1.0, 2.0, 3.0])
    p2 = Point(3, 1, [4.0, 5.0, 6.0])
    
    # Create point set
    ps = PointSet([p1, p2])
    
    # Add another point
    p3 = alloc_point(3, 2)
    p3.coord = [7.0, 8.0, 9.0]
    ps.add_point(p3)
    
    # Print using class methods
    print("Class method printing:")
    p1.print_point()
    ps.print_point_set()
    
    # Print using standalone functions
    print("\nStandalone function printing:")
    print_point(p2)
    print_point_set(ps)