# %%
import copy
import heapq
import numpy as np
from axiom import axiom

class MechanismMinMaxSolver():
    def __init__(self, profile, mechanism="greedy"):
        self.__profile = profile
        self.__costs = profile.costs
        self.__budget = profile.budget
        self.__approvals = profile.approvals
        self.__ballots = profile.ballots

        self.__projects = []
        self.__group_fraction = [sum(self.__profile.labels == cluster) / self.__profile.n_voters for cluster in range(max(self.__profile.labels)+1)]
        self.__group_ballot = [self.__ballots[self.__profile.labels == cluster].mean(0) for cluster in range(len(self.__group_fraction))]
        self.__group_gain_project = [ballot * self.__costs / self.__budget for ballot in self.__group_ballot]

        self.__mechanisms = {
            "greedy": self.__greedy_init,
            "random": self.__random_init
        }
        self.__mechanisms[mechanism]()

    def __random_init(self):
        # initialiseer the projects using greedy approval
        tmp = copy.copy(list(enumerate(self.__costs)))
        np.random.shuffle(tmp)
        for id, cost in tmp:
            if self.__budget >= cost:
                self.__projects.append(id)
                self.__budget -= cost

    def __greedy_init(self):
        # initialiseer the projects using greedy approval
        tmp = enumerate(zip(self.__costs, self.__approvals))
        for id, (cost, _) in sorted(tmp, key=lambda x: (x[1][1], x[1][0], x[0]), reverse=True):
            if self.__budget >= cost:
                self.__projects.append(id)
                self.__budget -= cost

    def __min_max_equitability(self):
        self.__max_cluster = {i:0 for i in range(len(self.__group_fraction))}
        for _ in range(5000):
            self.__optimise_for_cluster(*self.__get_max_equitability_cluster())
            if 3 in self.__max_cluster.values():
                if 0 not in self.__max_cluster.values():
                    break
                for c in self.__max_cluster:
                    self.__max_cluster[c] -= 1

        return self.__projects

    def __max_equitability(self):
        current_gain = [ggp[self.__projects].sum() for ggp in self.__group_gain_project]
        current_gain /= sum(current_gain)
        return np.max(np.abs(current_gain - self.__group_fraction))

    def __get_max_equitability_cluster(self):
        current_gain = [ggp[self.__projects].sum() for ggp in self.__group_gain_project]
        current_gain /= sum(current_gain)
        score = current_gain - self.__group_fraction
        for cluster, value in self.__max_cluster.items():
            if value != 0:
                score[cluster] = 0
        cluster = np.argmax(np.abs(score))
        return cluster, score[cluster]

    def __get_feasible_projects(self):
        return [i for i in range(self.__profile.n_projects) if i not in self.__projects and self.__costs[i] <= self.__budget]

    def __get_feasible_set(self, n=0, last_added_project=-1):
        projects = self.__get_feasible_projects()

        if not projects:
            yield

        for p1 in projects:
            if last_added_project > p1:
                continue
            self.__projects.append(p1)
            self.__budget -= self.__costs[p1]
            yield from self.__get_feasible_set(n+1, p1)
            p = self.__projects.pop(-1)
            self.__budget += self.__costs[p1]

    def __optimise_for_cluster(self, cluster, score):
        s = list(zip(*sorted(enumerate(np.abs(self.__group_gain_project[cluster][self.__projects] - score)), key=lambda x: x[1])))[0]
        re_id1, re_id2 = np.random.choice(s[:3], 2, replace=False)
        re_id1, re_id2 = sorted([re_id1, re_id2], reverse=True)
        remove_project1 = self.__projects.pop(re_id1)
        remove_project2 = self.__projects.pop(re_id2)
        self.__budget += self.__costs[remove_project1] + self.__costs[remove_project2]

        new_project = []
        for _ in self.__get_feasible_set():
            new_project.append((axiom(self.__profile, self.__projects), tuple(self.__projects)))

        new_projects = []
        for id in sorted(new_project)[0][1]:
            if id in self.__projects:
                continue
            self.__projects.append(id)
            self.__budget -= self.__costs[id]
            new_projects.append(id)

        # nothing changed
        if set(new_projects) == set([remove_project1, remove_project2]):
            self.__max_cluster[cluster] += 1
        # else:
        #     print(self.__projects, cluster, score, new_projects, remove_project1, remove_project2)

    def solve(self):
        return self.__min_max_equitability()

    def __call__(self):
        return self.solve()

class MechanismMinMaxSolver2():
    def __init__(self, profile):
        self.__profile = profile
        self.__costs = profile.costs
        self.__budget = profile.budget
        self.__approvals = profile.approvals
        self.__ballots = profile.ballots

    def __min_max_equitability(self):
        self.__projects = []
        self.__group_fraction = [sum(self.__profile.labels == cluster) / self.__profile.n_voters for cluster in range(max(self.__profile.labels)+1)]
        self.__group_ballot = [self.__ballots[self.__profile.labels == cluster].mean(0) for cluster in range(len(self.__group_fraction))]
        self.__group_gain_project = [ballot * self.__costs / self.__budget for ballot in self.__group_ballot]

        #initialiseer the projects using greedy approval
        tmp = enumerate(zip(self.__costs, self.__approvals))
        for id, (cost, _) in sorted(tmp, key=lambda x: (x[1][1], x[1][0], x[0]), reverse=True):
            if self.__budget >= cost:
                self.__projects.append(id)
                self.__budget -= cost

        self.__max_cluster = {i:0 for i in range(len(self.__group_fraction))}
        for _ in range(10000):
            self.__optimise_for_cluster(*self.__get_max_equitability_cluster())
            if 3 in self.__max_cluster.values():
                if 0 not in self.__max_cluster.values():
                    break
                for c in self.__max_cluster:
                    self.__max_cluster[c] -= 1

        return self.__projects

    def __max_equitability(self):
        current_gain = [ggp[self.__projects].sum() for ggp in self.__group_gain_project]
        current_gain /= sum(current_gain)
        return np.max(np.abs(current_gain - self.__group_fraction))

    def __get_max_equitability_cluster(self):
        current_gain = [ggp[self.__projects].sum() for ggp in self.__group_gain_project]
        current_gain /= sum(current_gain)
        score = current_gain - self.__group_fraction
        for cluster, value in self.__max_cluster.items():
            if value != 0:
                score[cluster] = 0
        cluster = np.argmax(np.abs(score))
        return cluster, score[cluster]

    def __get_feasible_projects(self):
        return [i for i in range(self.__profile.n_projects) if i not in self.__projects and self.__costs[i] <= self.__budget]

    def __optimise_for_cluster(self, cluster, score):
        # cluster gets to much
        if score > 0:
            remove_project_id = np.argmin(np.abs(self.__group_gain_project[cluster][self.__projects] - score))
        else:
            # check this
            remove_project_id = np.argmax(np.abs(self.__group_gain_project[cluster][self.__projects] - score))

        remove_project = self.__projects.pop(remove_project_id)
        self.__budget += self.__costs[remove_project]

        # while self.__get_feasible_projects():
        new_project = []
        for project_id in self.__get_feasible_projects():
            self.__projects.append(project_id)
            new_project.append((axiom(self.__profile, self.__projects), project_id))
            del self.__projects[-1]

        best_project = sorted(new_project)[0][1]
        self.__projects.append(best_project)
        self.__budget -= self.__costs[best_project]

        # nothing changed
        if best_project == remove_project:
            self.__max_cluster[cluster] += 1
        else:
            print(self.__projects, cluster, score, best_project, remove_project)

    def solve(self):
        return self.__min_max_equitability()

    def __call__(self):
        return self.solve()

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
        self.__approvals = profile.approvals.astype(int)


    def solve(self):
        return self.__max_approval()


    def __max_approval(self):
        b = dict()
        sets = dict()
        for k in range(len(self.__costs)):
            for t in range(sum(self.__approvals)):
                if t == self.__approvals[k] and self.__costs[k] <= self.__budget and self.__costs[k] < b.get((k, t), np.inf):
                    b[(k, t)] = self.__costs[k]
                    sets[(k, t)] = {k}
                    continue
                prev_k = (k - 1, t)
                prev_kt = (k - 1, t - self.__approvals[k])
                p = b.get(prev_k, np.inf)
                q = b.get(prev_kt, np.inf) + self.__costs[k]
                if p < q and p <= self.__budget:
                    b[(k, t)] = p
                    sets[(k, t)] = sets[(prev_k)]
                elif q <= self.__budget:
                    b[(k, t)] = q
                    sets[(k, t)] = set(sets[(prev_kt)]).union({k})

        candidates = sorted(b.items(), key=lambda x: (x[0][1], x[1]), reverse=True)
        for candidate, cost in candidates:
            if cost <= self.__budget:
                break
        else:
            return []
        return sorted(list(sets[candidate]))


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

        while paths:
            current_path = heapq.heappop(paths)
            if current_path.remaining_budget == 0:
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
