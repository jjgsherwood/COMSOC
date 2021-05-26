import numpy as np
from scipy.spatial.distance import jensenshannon

def axiom(projects, profile, pr=False):
    ballots, labels, costs, budget = profile.ballots, profile.labels, profile.costs, profile.budget
    n_clusters = max(labels) + 1
    clusters = [ballots[labels == i] for i in range(n_clusters)]
    b = []
    r = []
    for cluster in clusters:
        b.append(np.sum(np.mean(cluster,0)[projects]*costs[projects]))
        r.append(len(cluster)/len(ballots))
    b = np.array(b)
    b /= sum(b)

    # # Hellinger distance
    # return np.sqrt(1 - np.sum(np.sqrt(b * r)))

    return jensenshannon(b,r)

if __name__ == '__main__':
    from approval_profile import *
    from mechanism import *
    from cluster import *
    import time


    # profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
    profile = Profile_Synthetic(list(range(4000, 1, -200)), list(range(1000, 1, -50)), budget_distribution=uniform, low=500, high=10000, spread_of_approvals=2.5, sdcavpd=0.3, noise=0.02)
    # profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)

    label_profile(profile)

    mechanism = Mechanism(profile)
    t = time.process_time()
    projects = mechanism.solve("max_approval_DP")
    print(f"max approval DP took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    t = time.process_time()
    projects = mechanism.solve()
    print(f"Greedy took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    t = time.process_time()
    projects = mechanism.solve('max_approval')
    print(f"max approval took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
