# %%
import numpy as np

from collections import defaultdict

def MA(cost, approval):
    b = dict()

    for k in range(len(cost)):
        for t in range(sum(approval)):
            if k == 0:
                if t == approval[k]:
                    b[(k, t)] = cost[k]
            p = b.get((k - 1, t), np.inf)
            q = b.get((k - 1, t - approval[k] + cost[k]), np.inf)
            b_min = min(p, q)
            if b_min < np.inf:
                b[(k, t)] = b_min

    print(b)
                

if __name__ == "__main__":
    costs = [5,1,2,4,6,2]
    approvals = [33,32,4,10,9,21]

    MA(costs, approvals)

# %%
