
class FeasibleSets():
    """
    Class for computing feasible subsets. 
    """

    def __init__(self, projects, costs, budget, approvals):
        self.__projects = projects
        self.__costs = costs
        self.__budget = budget

        self.feasible_sets = dict()
