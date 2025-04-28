import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple
from point import Point

class GreedyAlgorithms:
    @staticmethod
    def run_greedy(dataP: List[Point], r: int) -> Tuple[List[Point], float]:
        """Greedy algorithm for regret minimization"""
        
        start_time = time.time()
        dim = dataP[0].dimension if dataP else 0
        R = []
        
        if not dataP or r < 1:
            return [], 0.0

        # Find initial point with maximum first coordinate
        max_index = max(range(len(dataP)), key=lambda i: dataP[i].coordinates[0])
        R.append(dataP[max_index])
        selected_indices = [max_index]

        # Create Gurobi environment
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        while len(selected_indices) < r:
            max_regret = 0.0
            best_index = -1
            
            for i in range(len(dataP)):
                if i in selected_indices:
                    continue
                
                # Create optimization model
                with gp.Model(env=env) as model:
                    model.setParam('OutputFlag', 0)
                    
                    # Create variables
                    X = model.addVars(dim + 1, lb=0.0, ub=GRB.INFINITY, name="X")
                    model.update()
                    
                    # Set objective
                    model.setObjective(X[dim], GRB.MAXIMIZE)
                    
                    # Add equality constraint
                    expr = gp.LinExpr()
                    for j in range(dim):
                        expr += dataP[i].coordinates[j] * X[j]
                    model.addConstr(expr == 1.0, "eq_constraint")
                    
                    # Add inequality constraints
                    for p in R:
                        ineq_expr = gp.LinExpr()
                        ineq_expr += -X[dim]
                        for j in range(dim):
                            ineq_expr += (dataP[i].coordinates[j] - p.coordinates[j]) * X[j]
                        model.addConstr(ineq_expr >= 0, f"ineq_{i}_{len(R)}")
                    
                    # Solve optimization
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL:
                        current_regret = X[dim].X
                        if current_regret > max_regret:
                            max_regret = current_regret
                            best_index = i

            if best_index == -1:
                break

            selected_indices.append(best_index)
            R.append(dataP[best_index])

        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        return R[:r], elapsed

    @staticmethod
    def run_matroid_greedy(dataP: List[Point], r: int, group_id: int,  # REMOVED k
                          grouped_data: Dict[int, List[Point]],
                          fairness_constraints: Dict[int, Tuple[int, int, int]]) -> Tuple[List[Point], float]:
        """Matroid-constrained greedy algorithm with fairness constraints"""
    
        start_time = time.time()
        dim = dataP[0].dimension if dataP else 0
        R = []
        selected_indices = []
        fairness_counts = {g: 0 for g in fairness_constraints}

        if not dataP or r < 1:
            return [], 0.0

        # Find initial point with maximum first coordinate
        max_index = max(range(len(dataP)), key=lambda i: dataP[i].coordinates[0])
        selected_indices.append(max_index)
        R.append(dataP[max_index])
        group = dataP[max_index].get_category(group_id)
        fairness_counts[group] += 1

        # Create Gurobi environment
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        candidates = GreedyAlgorithms._construct_candidates(grouped_data, fairness_constraints, 
                                                          selected_indices, fairness_counts, r)

        while len(selected_indices) < r and candidates:
            max_regret = 0.0
            best_index = -1

            for i in candidates:
                if i in selected_indices:
                    continue
                
                # Create optimization model
                with gp.Model(env=env) as model:
                    model.setParam('OutputFlag', 0)
                    
                    # Create variables
                    X = model.addVars(dim + 1, lb=0.0, ub=GRB.INFINITY, name="X")
                    model.update()
                    
                    # Set objective
                    model.setObjective(X[dim], GRB.MAXIMIZE)
                    
                    # Add equality constraint
                    expr = gp.LinExpr()
                    for j in range(dim):
                        expr += dataP[i].coordinates[j] * X[j]
                    model.addConstr(expr == 1.0, "eq_constraint")
                    
                    # Add inequality constraints
                    for p in R:
                        ineq_expr = gp.LinExpr()
                        ineq_expr += -X[dim]
                        for j in range(dim):
                            ineq_expr += (dataP[i].coordinates[j] - p.coordinates[j]) * X[j]
                        model.addConstr(ineq_expr >= 0, f"ineq_{i}_{len(R)}")
                    
                    # Solve optimization
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL:
                        current_regret = X[dim].X
                        if current_regret > max_regret:
                            max_regret = current_regret
                            best_index = i

            if best_index != -1 and best_index not in selected_indices:
                selected_indices.append(best_index)
                R.append(dataP[best_index])
                group = dataP[best_index].get_category(group_id)
                fairness_counts[group] += 1
                candidates = GreedyAlgorithms._construct_candidates(grouped_data, fairness_constraints,
                                                                  selected_indices, fairness_counts, r)
            else:
                # Fallback to random selection from candidates
                if candidates:
                    best_index = np.random.choice(candidates)
                    selected_indices.append(best_index)
                    R.append(dataP[best_index])
                    group = dataP[best_index].get_category(group_id)
                    fairness_counts[group] += 1
                    candidates = GreedyAlgorithms._construct_candidates(grouped_data, fairness_constraints,
                                                                      selected_indices, fairness_counts, r)

        elapsed = (time.time() - start_time) * 1000
        return R[:r], elapsed

    @staticmethod
    def _construct_candidates(grouped_data: Dict[int, List[Point]],
                             fairness_constraints: Dict[int, Tuple[int, int, int]],
                             selected_indices: List[int],
                             fairness_counts: Dict[int, int],
                             r: int) -> List[int]:
        candidates = []
        remaining = r - len(selected_indices)
        
        for group, (lc, uc, ki) in fairness_constraints.items():
            current_count = fairness_counts.get(group, 0)
            max_additional = min(uc - current_count, remaining)
            
            if max_additional > 0:
                group_points = [i for p in grouped_data[group] 
                              if (i := p.id) not in selected_indices]
                candidates.extend(group_points[:max_additional])
        
        return list(set(candidates))