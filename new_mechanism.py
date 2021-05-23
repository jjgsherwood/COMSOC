import numpy as np

def min_max_equitability(profile):
    get_feasible_projects = lambda projects, n_projects, costs, budget: [i for i in range(n_projects) if i not in projects and costs[i] <= budget]
    projects = []
    costs = profile.costs
    budget = profile.budget

    group_fraction = [sum(profile.labels == cluster) / profile.n_voters for cluster in range(max(profile.labels)+1)]
    group_ballot = [profile.ballots[profile.labels == cluster].mean(0) for cluster in range(len(group_fraction))]
    group_gain_project = [ballot * costs / profile.budget for ballot in group_ballot]

    for id, cost in enumerate(costs):
        if budget >= cost:
            projects.append(id)
            budget -= cost

    return projects


if __name__ == '__main__':
    from approval_profile import *
    from mechanism import *
    from cluster import *
    from axiom import *
    import time

    try:
        profile = Profile_Synthetic.load("test.pb")
    except:
        profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
        # profile = Profile_Synthetic(list(range(4000, 1, -200)), list(range(1000, 1, -50)), budget_distribution=uniform, low=500, high=10000, spread_of_approvals=2.5, sdcavpd=0.3, noise=0.02)
        # profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)

    label_profile(profile)
    # profile.save("test2.pb")

    t = time.process_time()
    projects = min_max_equitability(profile)
    print(f"greedy equitability took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    print()

    mechanism = Mechanism(profile)
    t = time.process_time()
    projects = mechanism.solve("max_approval_DP")
    print(f"max approval DP took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    print()
    t = time.process_time()
    projects = mechanism.solve()
    print(f"Greedy took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    print()
