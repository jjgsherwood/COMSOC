# %%
import numpy as np

from collections import defaultdict


def max_approval(costs, approvals, budget):
    candidates = get_candidates(costs, approvals)
    return get_optimal_solution(candidates, approvals, budget)


def get_candidates(costs, approvals):
    b = dict()
    for k in range(len(costs)):
        for t in range(sum(approvals)):
            if t == approvals[k]:
                b[(k, t)] = costs[k]
            p = b.get((k - 1, t), np.inf)
            q = b.get((k - 1, t - approvals[k]), np.inf) + costs[k]
            print("k", k, "t", t, "budgets", p, q)
            b_min = min(p, q)
            if b_min < np.inf:
                b[(k, t)] = b_min
    return sorted(b.items(), key=lambda x: (x[0][1], x[1]), reverse=True)


def get_optimal_solution(candidates, approvals, budget):
    solution, viable_candidates = set(), set()
    optimal_not_found = True
    optimal_candidate = None

    for candidate, cost in candidates:
        if cost <= budget and optimal_not_found:
            optimal_candidate = candidate
            optimal_not_found = False
        viable_candidates.add(candidate)
    
    if optimal_candidate is None:
        return None

    item, approval = optimal_candidate

    # reconstruct solution
    while approval > 0:
        while (item - 1, approval) in viable_candidates:
            item -= 1
        solution.add(item)
        approval -= approvals[item]
    return solution
        

if __name__ == "__main__":
    costs = [11, 7, 10, 2]
    approvals = [33,32,4,10]
    budget = 10

    solution = max_approval(costs, approvals, budget)
    print(solution)

# %%

# %%
