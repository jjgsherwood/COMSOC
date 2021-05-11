# %%
import numpy as np

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
            "max_approval": self.__max_approval,
            "greedy_approval": self.__greedy_approval
        }


    def __repr__(self):
        return str(self.__profile)


    def __max_approval(self):
        raise NotImplementedError()


    def __greedy_approval(self):
        solution = []
        projects = enumerate(zip(self.__profile.projects, self.__profile.approvals))
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
    from approval_profile import Profile

    profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
    mechanism = Mechanism(profile)

# %%
