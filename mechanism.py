# %%
import numpy as np
from mechanism_solver import MechanismAStarSolver, MechanismDynamicSolver, MaxApprovalSolver
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
            "greedy_approval": self.__greedy_approval
        }


    def __repr__(self):
        return str(self.__profile)


    def __max_approval_DP(self):
        return MaxApprovalSolver(self.__profile)()


    def __max_approval(self):
        return MechanismAStarSolver(self.__profile, 'max_approval')()


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


    def solve(self, mechanism="greedy_approval"):
        try:
            projects = self.__mechanisms[mechanism]()
        except KeyError:
            raise ValueError(f"Mechanism: unknown mechanism '{mechanism}.'")

        return projects


if __name__ == '__main__':
    from approval_profile import *
    import time
    from MADP import max_approval

    # profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
    profile = Profile_Synthetic(list(range(1100, 100, -110)), list(range(250, 10, -30)), budget_distribution=uniform, low=500, high=10000)
    # profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)
    mechanism = Mechanism(profile)
    t = time.process_time()
    projects = mechanism.solve("max_approval_DP")
    print(f"max approval DP took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    t = time.process_time()
    projects = mechanism.solve()
    print(f"Greedy took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
    t = time.process_time()
    projects = mechanism.solve('max_approval')
    print(f"max approval took {-t + time.process_time()}")
    print(projects)
    print("approval:", profile.get_approval_percentage(projects))
    print("budget:", profile.get_budget_percentage(projects))
# %%
