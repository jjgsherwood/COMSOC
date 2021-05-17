# %%
import copy
import heapq
import numpy as np

class MechanismDynamicSolver():
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
        self.__dict = {}
        self.__i = 0

    def KS(self, n, budget):
        self.__i += 1
        if not self.__i % 100:
            print(self.__i)
        if n == 0 or budget == 0:
            return 0, []

        try:
            tmp = self.__dict[(n, budget)]
        except KeyError:
            if self.__costs[n] > budget:
                tmp = self.KS(n-1, budget)
                self.__dict[(n-1, budget)] = tmp
            else:
                tmp1 = self.KS(n-1, budget)
                self.__dict[(n-1, budget)] = tmp1

                gain, projects = self.KS(n-1, budget - self.__costs[n])
                gain += self.__approvals[n]
                projects += [n]
                tmp2 = (gain, projects)
                self.__dict[(n-1, budget - self.__costs[n])] = tmp2
                tmp = max([tmp1,tmp2], key=lambda x: x[0])
        return tmp

    def solve(self):
        return self.KS(len(self.__costs)-1, self.__budget)[1]

    def __call__(self):
        return self.solve()

class MaxApprovalSolver():
    def __init__(self, profile):
        self.__costs = profile.costs
        self.__budget = profile.budget
        self.__approvals = profile.approvals


    def solve(self):
        candidates = self.__candidates()
        return self.__reconstruct(candidates)


    def __candidates(self):
        b = dict()
        for k in range(len(self.__costs)):
            for t in range(int(sum(self.__approvals))):
                if t == self.__approvals[k] and self.__costs[k] < b.get((k, t), np.inf):
                    b[(k, t)] = self.__costs[k]
                p = b.get((k - 1, t), np.inf)
                q = b.get((k - 1, t - self.__approvals[k]), np.inf) + self.__costs[k]
                b_min = min(p, q)
                if b_min < np.inf:
                    b[(k, t)] = b_min
        return sorted(b.items(), key=lambda x: (x[0][1], x[1]), reverse=True)


    def __reconstruct(self, candidates):
        solution, viable_candidates = list(), set()
        optimal_not_found = True
        optimal_candidate = None

        for candidate, cost in candidates:
            if cost <= self.__budget and optimal_not_found:
                print("candidate found:", candidate, cost)
                optimal_candidate = candidate
                optimal_not_found = False
            if cost <= self.__budget:
                viable_candidates.add(candidate)

        if optimal_candidate is None:
            return None

        item, approval = optimal_candidate

        # reconstruct solution
        while approval > 0:
            while (item - 1, approval) in viable_candidates:
                item -= 1
            solution.append(item)
            approval -= self.__approvals[item]
        return solution


    def __call__(self):
        return self.solve()

class MechanismAStarSolver():
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
        max_approval_per_budget = sorted(self.__approvals / self.__costs,reverse=True)[i-4]

        self.args = [max_approval_per_budget, self.__budget]
        self.Path = Path_Max_Approval


    def solve(self):
        start = self.Path(*self.args)
        # print('expected gain', start.expected_max_gain)
        # print('budget', self.__budget)
        # print('id, approval, cost, ratio')
        # for id, (cost, approval) in enumerate(zip(self.__costs, self.__approvals)):
        #     print(id, '\t', approval,'\t' ,cost, '\t' ,approval/cost)
        paths = [start.add_project(id, cost, approval) for id, (cost, approval) in enumerate(zip(self.__costs, self.__approvals))]
        paths.sort(reverse=False)
        evaluated_sets = set(paths)
        heapq.heapify(paths)

        i = 0
        while paths:
            current_path = heapq.heappop(paths)
            # i+=1
            # if not i % 100:
            #     print("iteration", i)
            if current_path.remaining_budget == 0:
                print(len(list(current_path.projects)))
                return list(current_path.projects)

            no_more_paths = True
            for id, (cost, approval) in enumerate(zip(self.__costs, self.__approvals)):
                if current_path.remaining_budget < cost or id in current_path.projects:
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
                heapq.heappush(paths, current_path)

        return None


    def __call__(self):
        return self.solve()

class Path():
    def __init__(self, budget, *args):
        self.projects = set()
        self.remaining_budget = budget
        self.current_gain = 0
        self.expected_max_gain = self.heuristic()
        self.newest_id = None


    def add_project(self, project_id, cost, approval):
        new = copy.deepcopy(self)
        new.projects.add(project_id)
        new.newest_id = project_id
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

# %%
