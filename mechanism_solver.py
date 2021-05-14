import copy
import heapq
import numpy as np

class MechanismSolver():
    """
    Class for computing feasible subsets.
    """

    def __init__(self, profile, mechanism):
        self.__profile = profile
        self.__costs = profile.costs
        self.__budget = profile.budget
        self.__approvals = profile.approvals

        self.__mechanisms = {
            "max_approval": self.__max_approval,
        }
        self.__mechanisms[mechanism]()

    def __max_approval(self):
        # find the minimum number of projects that can be approved
        budget = self.__budget
        for i, (_, cost) in enumerate(sorted(zip(self.__approvals / self.__costs, self.__costs), reverse=True)):
            budget -= cost
            if budget <= cost:
                break
        max_approval_per_budget = np.median(sorted(self.__approvals / self.__costs,reverse=True)[:i])

        self.args = [max_approval_per_budget, self.__budget]
        self.Path = Path_Max_Approval

    def solve(self):
        start = self.Path(*self.args)
        print('expected max gain', start.expected_max_gain)
        print('budget', self.__budget)
        # print('id, approval, cost, ratio')
        # for id, (cost, approval) in enumerate(zip(self.__costs, self.__approvals)):
        #     print(id, '\t', approval,'\t' ,cost, '\t' ,approval/cost)
        paths = [start.add_project(id, cost, approval) for id, (cost, approval) in enumerate(zip(self.__costs, self.__approvals))]
        paths.sort(reverse=False)
        evaluated_sets = set(paths)
        heapq.heapify(paths)
        while paths:
            current_path = heapq.heappop(paths)
            if current_path.remaining_budget == 0:
                return list(current_path.projects)

            no_more_paths = True
            for id, (cost, approval) in enumerate(zip(self.__costs, self.__approvals)):
                if id in current_path.projects or current_path.remaining_budget < cost:
                    continue

                no_more_paths = False
                new_path = current_path.add_project(id, cost, approval)
                if new_path in evaluated_sets:
                    continue
                heapq.heappush(paths, new_path)
                evaluated_sets.add(new_path)

            if no_more_paths:
                current_path.remaining_budget = 0
                current_path.expected_max_gain = 0
                heapq.heappush(paths, new_path)

        return None

    def __call__(self):
        return self.solve()

class Path():
    def __init__(self, budget, *args):
        self.projects = set()
        self.remaining_budget = budget
        self.current_gain = 0
        self.expected_max_gain = self.heuristic()

    def add_project(self, project_id, cost, approval):
        new = copy.deepcopy(self)
        new.projects.add(project_id)
        new.cost_fn(cost)
        new.gain_fn(project_id=project_id, cost=cost, approval=approval)
        new.expected_max_gain = new.heuristic(project_id=project_id, cost=cost, approval=approval)
        return new

    def cost_fn(self, cost):
        self.remaining_budget -= cost

    def heuristic(self, **kwargs):
        raise NotImplementedError

    def gain_fn(self, approval, **kwargs):
        raise NotImplementedError

    def __lt__(self, other):
        return self.current_gain + self.expected_max_gain >= other.current_gain + other.expected_max_gain

    def __eq__(self, other):
        return self.projects == other.projects

    def __hash__(self):
        return hash(tuple(sorted(self.projects)))

    def __repr__(self):
        return str(self.projects)

class Path_Max_Approval(Path):
    def __init__(self, max_approval_per_budget, *args):
        self.max_approval_per_budget = max_approval_per_budget
        super().__init__(*args)

    def heuristic(self, **kwargs):
        return self.remaining_budget * self.max_approval_per_budget

    def gain_fn(self, approval, **kwargs):
        self.current_gain += approval
