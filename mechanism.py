# %%
import numpy as np
from mechanism_solver import *
from approval_profile import uniform

class Mechanism():
    """
    Find project allocations from given Profile using different
    mechanisms.

    methods:
        - solve: find allocation. Defaults to greedy approval.

    Supported mechanisms:
        - max_approval.
        - greedy approval.
    """

    def __init__(self, profile):
        self.__profile = profile

        self.__mechanisms = {
            "max_approval_DP": self.__max_approval_DP,
            "max_approval": self.__max_approval,
            "greedy_approval": self.__greedy_approval,
            "greedy_equitability": self.__greedy_equitability,
            "min_max_equitability": self.__min_max_equitability,
            "n_random_min_max_equitability": self.__n_random_min_max_equitability
        }


    def __repr__(self):
        return str(self.__profile)


    def __max_approval_DP(self):
        return MaxApprovalSolver(self.__profile)()

    def __max_approval(self):
        return MechanismAStarSolver(self.__profile, 'max_approval')()

    def __min_max_equitability(self):
        return MechanismMinMaxSolver(self.__profile)()

    def __n_random_min_max_equitability(self):
        score = np.inf
        for _ in range(10):
            projects = MechanismMinMaxSolver(self.__profile, "random")()
            new_score = axiom(projects, self.__profile)
            if new_score < score:
                best_projects = projects
        return best_projects

    def __greedy_approval(self):
        solution = []
        projects = enumerate(zip(self.__profile.costs, self.__profile.approvals))
        budget = self.__profile.budget

        for project, (cost, _) in sorted(projects, key=lambda x: (x[1][1], x[1][0], x[0]), reverse=True):
            if budget - cost < 0:
                continue
            solution.append(project)
            budget -= cost

        return solution

    def __greedy_equitability(self):
        get_feasible_projects = lambda projects, n_projects, costs, budget: [i for i in range(n_projects) if i not in projects and costs[i] <= budget]
        projects = []
        costs = self.__profile.costs
        budget = self.__profile.budget

        group_fraction = [sum(self.__profile.labels == cluster) / self.__profile.n_voters for cluster in range(max(self.__profile.labels)+1)]
        group_ballot = [self.__profile.ballots[self.__profile.labels == cluster].mean(0) for cluster in range(len(group_fraction))]
        group_gain_project = [ballot * costs / budget for ballot in group_ballot]

        free_projects = get_feasible_projects(projects, self.__profile.n_projects, costs, budget)
        while free_projects:
            cluster = np.argmin([ggp[projects].sum() * gf for gf, ggp in zip(group_fraction, group_gain_project)])
            new_porject = free_projects[np.argmax(group_ballot[cluster][free_projects])]
            projects.append(new_porject)
            budget -= costs[new_porject]
            free_projects = get_feasible_projects(projects, self.__profile.n_projects, costs, budget)

        return projects


    def solve(self, mechanism="greedy_approval"):
        try:
            projects = self.__mechanisms[mechanism]()
        except KeyError:
            raise ValueError(f"Mechanism: unknown mechanism '{mechanism}.'")

        return projects


if __name__ == '__main__':
    from approval_profile import *
    from cluster import *
    from axiom import *
    import time

    try:
        profile = Profile_Synthetic.load("test.pb")
    except:
        # profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
        profile = Profile_Synthetic(list(range(2000, 1, -100)), list(range(1000, 1, -100)), budget_distribution=uniform, low=500, high=10000, spread_of_approvals=2.5, sdcavpd=0.3, noise=0.02)
        # profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)

    label_profile(profile)
    profile.save("test.pb")

    mechanism = Mechanism(profile)

    t = time.process_time()
    projects = mechanism.solve("n_random_min_max_equitability")
    print(f"n random min max equitability took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    print()

    t = time.process_time()
    projects = mechanism.solve("min_max_equitability")
    print(f"min max equitability took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    print("axiom score", axiom(projects, profile))
    print()

    t = time.process_time()
    projects = mechanism.solve("greedy_equitability")
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
# %%
