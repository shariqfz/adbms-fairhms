from pulp import *

class Point:
    def __init__(self, dim):
        self.dim = dim
        self.coord = [0.0] * dim

class PointSet:
    def __init__(self, num_points, dim):
        self.numberOfPoints = num_points
        self.points = [Point(dim) for _ in range(num_points)]

def worstDirection(s, pt, index=None):
    if index is not None:
        K = index
        points = s.points[:K]
    else:
        K = s.numberOfPoints
        points = s.points
    D = pt.dim
    epsilon = 1e-13  # Small epsilon to approximate equality

    prob = LpProblem("MaxRegretRatio", LpMaximize)

    # Variables
    v = [LpVariable(f"v_{j}", lowBound=0) for j in range(D)]
    x = LpVariable("x")  # Unbounded by default

    # Objective
    prob += x

    # Constraints for each point in the subset
    for i in range(K):
        s_i = points[i]
        constraint = lpSum((s_i.coord[j] - pt.coord[j]) * v[j] for j in range(D)) + x <= 0
        prob += constraint, f"q_{i+1}"

    # r1 and r2 constraints
    sum_pt_v = lpSum(pt.coord[j] * v[j] for j in range(D))
    prob += (sum_pt_v <= 1 + epsilon), "r1"
    prob += (sum_pt_v >= 1 - epsilon), "r2"

    # Solve
    prob.solve()

    if LpStatus[prob.status] != 'Optimal':
        raise ValueError("LP problem did not solve optimally.")

    regret_ratio = x.value()
    v_values = [var.value() for var in v]

    return (regret_ratio, v_values)

def evaluateLP(p, S, VERBOSE=False):
    max_regret = 0.0
    D = p.points[0].dim

    for pt in p.points:
        (regret_ratio, v) = worstDirection(S, pt)

        # Compute maxN and maxK
        maxN = max(sum(p.coord[j] * v[j] for j in range(D)) for p in p.points)
        maxK = max(sum(s.coord[j] * v[j] for j in range(D)) for s in S.points)

        if maxN == 0:
            current_regret = 1.0
        else:
            current_regret = 1.0 - (maxK / maxN)

        if current_regret > max_regret:
            max_regret = current_regret

    if VERBOSE:
        print(f"LP max regret ratio = {max_regret}")

    return max_regret

# Test cases
def test1():
    pt = Point(2)
    pt.coord = [0.9, 0.9]
    s = PointSet(2, 2)
    s.points[0].coord = [1.0, 0.0]
    s.points[1].coord = [0.0, 1.0]
    (regret, v) = worstDirection(s, pt)
    print("Test 1:")
    print(f"Regret ratio: {regret}")
    print(f"Direction v: {v}")

def test2():
    pt = Point(2)
    pt.coord = [0.4, 0.4]
    s = PointSet(2, 2)
    s.points[0].coord = [1.0, 0.0]
    s.points[1].coord = [0.0, 1.0]
    (regret, v) = worstDirection(s, pt)
    print("Test 2:")
    print(f"Regret ratio: {regret}")
    print(f"Direction v: {v}")

if __name__ == "__main__":
    test1()
    test2()