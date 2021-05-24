import numpy as np




if __name__ == '__main__':
    from approval_profile import *
    from mechanism import *
    from cluster import *
    from axiom import *
    import time

    try:
        profile = Profile_Synthetic.load("test1.pb")
    except:
        # profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
        profile = Profile_Synthetic(list(range(2000, 1, -100)), list(range(1000, 1, -100)), budget_distribution=uniform, low=500, high=10000, spread_of_approvals=2.5, sdcavpd=0.3, noise=0.02)
        # profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)

    label_profile(profile)
    # profile.save("test.pb")
    mechanism = Mechanism(profile)

    t = time.process_time()
    projects = mechanism.solve("min_max_equitability")
    print(f"greedy equitability took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    print()

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
