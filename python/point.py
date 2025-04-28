import numpy as np

class Point:
    def __init__(self, dimension=0, id=-1, coordinates=None, categories=None):
        if coordinates is not None:
            self.coordinates = np.array(coordinates, dtype=np.float64)
            self.dimension = len(self.coordinates)
            if dimension != 0 and dimension != self.dimension:
                raise ValueError("Dimension does not match coordinates length")
        else:
            self.dimension = dimension
            self.coordinates = np.zeros(dimension, dtype=np.float64)
        self.id = id
        self.categories = list(categories) if categories is not None else []

    def __sub__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for subtraction")
        return Point(coordinates=self.coordinates - other.coordinates)

    def __mul__(self, factor):
        return Point(coordinates=self.coordinates * factor)
    
    def dot_product(self, other: 'Point') -> float:
        """Compute dot product with another Point"""
        return sum(a * b for a, b in zip(self.coordinates, other.coordinates))

    def get_dimension(self):
        return self.dimension

    def get_id(self):
        return self.id

    def get_coordinate(self, idx):
        if idx < 0 or idx >= self.dimension:
            raise IndexError("Index out of bounds")
        return self.coordinates[idx]

    def distance_to(self, other):
        return np.linalg.norm(self.coordinates - other.coordinates)

    def length(self):
        return np.linalg.norm(self.coordinates)

    def dotP(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for dot product")
        return np.dot(self.coordinates, other.coordinates)

    def dominates(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for domination check")
        return (np.all(self.coordinates >= other.coordinates) and 
                np.any(self.coordinates > other.coordinates))

    def dump(self, prefix="", suffix=""):
        formatted_coords = ['%.6f' % x for x in self.coordinates]
        print(f"{prefix}[ {', '.join(formatted_coords)} ]{suffix}")

    def scale_to_length(self, new_length):
        current_length = self.length()
        if current_length == 0:
            if new_length != 0:
                raise ValueError("Cannot scale a zero-length vector to non-zero length")
            return
        scaling_factor = new_length / current_length
        self.coordinates *= scaling_factor

    @staticmethod
    def abs(other_point):
        return Point(coordinates=np.abs(other_point.coordinates))

    @staticmethod
    def prod(M, other_point):
        result = np.dot(M, other_point.coordinates)
        return Point(coordinates=result)

    @classmethod
    def from_ublas(cls, ublas_vec):
        return cls(coordinates=ublas_vec)

    @staticmethod
    def to_ublas(p):
        return p.coordinates.copy()

    def print(self):
        formatted_coords = ['%.6f' % x for x in self.coordinates]
        print(f"dim={self.dimension}  id={self.id}  {' '.join(formatted_coords)}")

    def get_all_coords(self):
        return self.coordinates.tolist()

    def dotP_vector(self, coord):
        return np.dot(self.coordinates, np.array(coord, dtype=np.float64))

    def get_category(self, cate_id):
        return self.categories[int(cate_id)]

    def get_category_dim(self):
        return len(self.categories)

    def __repr__(self):
        return f"Point(dimension={self.dimension}, id={self.id}, coordinates={self.coordinates}, categories={self.categories})"
    
    @staticmethod
    def random_sphere(dim: int) -> 'Point':
        """Generate a random point on a unit sphere."""
        vec = np.random.normal(size=dim)
        vec /= np.linalg.norm(vec)
        return Point(dimension=dim, coordinates=vec)