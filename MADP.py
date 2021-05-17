# %%
import numpy as np


# def max_approval(costs, approvals, budget):
#     candidates = get_candidates(costs, approvals, budget)
#     return get_optimal_solution(candidates, approvals, budget)


def max_approval(costs, approvals, budget):
    b = dict()
    sets = dict()
    for k in range(len(costs)):
        print(k)
        for t in range(sum(approvals)):
            if t == approvals[k] and costs[k] <= budget and costs[k] < b.get((k, t), np.inf):
                b[(k, t)] = costs[k]
                sets[(k, t)] = {k}
                continue
            prev_k = (k - 1, t)
            prev_kt = (k - 1, t - approvals[k])
            p = b.get(prev_k, np.inf)
            q = b.get(prev_kt, np.inf) + costs[k]
            if p < q and p <= budget:
                b[(k, t)] = p
                sets[(k, t)] = sets[(prev_k)]
            elif q <= budget:
                b[(k, t)] = q
                sets[(k, t)] = set(sets[(prev_kt)]).union({k})
    
    candidates = sorted(b.items(), key=lambda x: (x[0][1], x[1]), reverse=True)
    for candidate, cost in candidates:
        if cost <= budget:
            break
    else:
        return []
    return sorted(list(sets[candidate]))


# def get_optimal_solution(candidates, approvals, budget):
#     solution, viable_candidates = set(), set()
#     optimal_not_found = True
#     optimal_candidate = None

#     for candidate, cost in candidates:
#         if cost <= budget and optimal_not_found:
#             optimal_candidate = candidate
#             optimal_not_found = False
#         viable_candidates.add(candidate)
    
#     if optimal_candidate is None:
#         return None

#     item, approval = optimal_candidate

#     # reconstruct solution
#     while approval > 0:
#         while (item - 1, approval) in viable_candidates:
#             item -= 1
#         solution.add(item)
#         approval -= approvals[item]
#     return solution
        

if __name__ == "__main__":
    from approval_profile import *
    costs = [11, 7, 10, 2]
    approvals = [33,32,4,10]
    budget = 10

    profile = Profile_Synthetic()
    profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)


    projects = max_approval(profile.costs, profile.approvals.astype(int), profile.budget)
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))


# %%
