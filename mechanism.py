# %%
import numpy as np

class Mechanism():

    def __init__(self, profile):
        self.__profile = profile

        self.__mechanisms = {
            "max_approval": self.__max_approval,
            "greedy_approval": self.__greedy_approval
        }


    def __max_approval(self):
        raise NotImplementedError()


    def __greedy_approval(self):
        raise NotImplementedError()


    def solve(self, mechanism="greedy_approval"):
        try:
            return self.__mechanisms[mechanism]()
        except KeyError():
            raise ValueError(f"Mechanism: unknown mechanism '{mechanism}.'")

    
    def __repr__(self):
        return self.__profile




if __name__ == '__main__':
    from approval_profile import Profile

    test = Mechanism(Profile("data/poland_warszawa_2018_praga-poludnie.pb"))


# %%
