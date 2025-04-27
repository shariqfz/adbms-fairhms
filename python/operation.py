import math
import itertools
from itertools import product

EPSILON = 1e-9

class Point:
    def __init__(self, dim, id=-1):
        self.dim = dim
        self.coord = [0.0] * dim
        self.id = id

    def __repr__(self):
        return f"Point(id={self.id}, coord={self.coord})"

class PointSet:
    def __init__(self, num_points, dim=0):
        self.numberOfPoints = num_points
        self.points = [Point(dim) for _ in range(num_points)]

def is_zero(x):
    return abs(x) < EPSILON

def rand_f(min_v, max_v):
    import random
    return random.uniform(min_v, max_v)

def calc_dist(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1.coord, p2.coord)))

def calc_len(p):
    return math.sqrt(sum(a**2 for a in p.coord))

def copy_point(p):
    new_p = Point(p.dim, p.id)
    new_p.coord = list(p.coord)
    return new_p

def dot_prod(p1, p2):
    return sum(a * b for a, b in zip(p1.coord, p2.coord))

def dot_prod_v(p, v):
    return sum(a * b for a, b in zip(p.coord, v))

def sub_points(p1, p2):
    res = Point(p1.dim)
    res.coord = [a - b for a, b in zip(p1.coord, p2.coord)]
    return res

def add_points(p1, p2):
    res = Point(p1.dim)
    res.coord = [a + b for a, b in zip(p1.coord, p2.coord)]
    return res

def scale_point(c, p):
    res = Point(p.dim)
    res.coord = [c * a for a in p.coord]
    return res

def is_violated(normal_q, normal_p, e):
    if is_zero(calc_dist(normal_q, normal_p)):
        return True
    temp_normal = sub_points(normal_q, normal_p)
    temp = sub_points(e, normal_p)
    dp = dot_prod(temp_normal, temp)
    return dp > 0 and not is_zero(dp)

def max_point(point_set, v):
    max_val = -float('inf')
    max_point = None
    for p in point_set.points:
        current = dot_prod_v(p, v)
        if current > max_val:
            max_val = current
            max_point = p
    return max_point

def gauss_elimination(A):
    n = len(A)
    if n == 0:
        return []
    d = len(A[0]) - 1
    for i in range(d):
        max_row = i
        for k in range(i, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        A[i], A[max_row] = A[max_row], A[i]
        for k in range(i+1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, d+1):
                if j == i:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]
    x = [0] * d
    for i in reversed(range(d)):
        x[i] = A[i][d] / A[i][i]
        for k in reversed(range(i)):
            A[k][d] -= A[k][i] * x[i]
    return x

def project_onto_affine(space, p):
    if space.numberOfPoints == 0:
        raise ValueError("Empty space")
    if space.numberOfPoints == 1:
        return copy_point(space.points[0])
    
    dim = space.points[0].dim
    n = space.numberOfPoints - 1
    dir_vecs = [sub_points(space.points[i+1], space.points[0]) for i in range(n)]
    
    for i in range(1, n):
        for j in range(i):
            c = dot_prod(dir_vecs[i], dir_vecs[j]) / dot_prod(dir_vecs[j], dir_vecs[j])
            for k in range(dim):
                dir_vecs[i].coord[k] -= c * dir_vecs[j].coord[k]
    
    for vec in dir_vecs:
        norm = calc_len(vec)
        if norm > EPSILON:
            for k in range(dim):
                vec.coord[k] /= norm
    
    tmp = sub_points(p, space.points[0])
    coord = [dot_prod(tmp, vec) for vec in dir_vecs]
    
    proj_coord = [0.0] * dim
    for j in range(n):
        for k in range(dim):
            proj_coord[k] += coord[j] * dir_vecs[j].coord[k]
    
    proj = add_points(space.points[0], Point(dim, coord=proj_coord))
    return proj

def build_input(t, dim):
    dist_bet = 1.0 / (t + 1)
    centers = [i * dist_bet + dist_bet / 2 for i in range(t + 1)]
    return [centers] * (dim - 1)

def cartesian_product(inputs):
    return list(itertools.product(*inputs))

def read_points(filename):
    with open(filename, 'r') as f:
        num_points, dim = map(int, f.readline().split())
        point_set = PointSet(num_points)
        for i in range(num_points):
            coords = list(map(float, f.readline().split()))
            p = Point(dim, i)
            p.coord = coords
            point_set.points[i] = p
    return point_set

def dominates(p1, p2):
    return all(a >= b for a, b in zip(p1.coord, p2.coord))

def skyline(p_set):
    skyline_points = []
    for p in p_set.points:
        dominated = False
        to_remove = []
        for i, sk_p in enumerate(skyline_points):
            if dominates(sk_p, p):
                dominated = True
                break
            if dominates(p, sk_p):
                to_remove.append(i)
        if not dominated:
            skyline_points = [sk_p for i, sk_p in enumerate(skyline_points) if i not in to_remove]
            skyline_points.append(p)
    result = PointSet(len(skyline_points))
    result.points = skyline_points
    return result

def insert_orth(points, count, v):
    dim = v.dim
    orth_num = 2**dim - 1
    points.append(v.coord.copy())
    count += 1
    for i in range(1, orth_num):
        bits = i
        new_point = [0.0] * dim
        for j in range(dim):
            if bits & 1:
                new_point[j] = v.coord[j]
            bits >>= 1
        points.append(new_point)
        count += 1
    return count