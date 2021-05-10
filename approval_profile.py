# %%

import numpy as np
import pickle
from pylab import rcParams


class Profile():
    def __init__(self, filename):
        self._metadata = {}
        self._projects = {}
        self._votes = {}

        # cant pickle file object
        with open(filename, "r", encoding="utf8") as f:
            self.__read_lines(f)

        self.__convert_projects()
        self.__convert_votes()


    def __repr__(self):
        return str(self._metadata)

        
    @property
    def ballots(self):
        return self._ballots


    @property
    def approvals(self):
        return sum(self._ballots)


    @property
    def budget(self):
        return float(self._metadata["budget"].replace(",", "."))

    
    @property
    def projects(self):
        return np.array([x[1] for x in sorted(self._projects.items(), key=lambda x: x[0])])

        
    def __convert_projects(self):
        self._projectid_to_index = {}
        tmp = {}
        for i, (proj_id, budget) in enumerate(self._projects.items()):
            self._projectid_to_index[proj_id] = i
            tmp[i] = budget
        self._projects = tmp

        
    def __convert_votes(self):
        self._votes = [np.array([self._projectid_to_index[x]]) if isinstance(x, int) else np.array([self._projectid_to_index[int(y)] for y in x.split(",")]) for x in self._votes.values()]
        self._ballots = np.zeros((self._metadata["num_votes"], self._metadata["num_projects"]))
        for i, vote in enumerate(self._votes):
            self._ballots[i,vote] = 1

       
    def __read_lines(self, f):
        _sections = {"META":self._metadata, 
                     "PROJECTS":self._projects, 
                     "VOTES":self._votes}
        _slices = {"key":"value",
                   "project_id":"cost",
                   "voter_id":"vote"}
        
        for line in f:
            line = line.strip()

            items = line.split(";")
            # find the right index for one of the properties (value, cost, vote)
            try:
                index = items.index(_slices[items[0]])
            except KeyError:
                pass
            else:
                continue 
            
            # switch to a new dict when a new section is found
            try: 
                _current = _sections[line]
            except KeyError:
                pass
            else:
                continue
            
            # read data
            try:
                try:
                    key = int(items[0])
                except ValueError:
                    key = items[0]
                _current[key] = int(items[index])
            except IndexError:
                pass
            except ValueError:
                _current[key] = items[index]


    def get_approval_percentage(self, projects):
        for votes in self._votes:
                    for project in projects:
                        if project in votes:
                            approvals += 1
                            break

        return approvals / len(self._votes)


    def get_cost(self, projects):
        return sum(self.projects[projects])


    def get_budget_percentage(self, projects):
        return self.get_cost(projects) / self.budget
        

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == '__main__':
    path = "data/profiles/"
    test = Profile("data/poland_warszawa_2018_praga-poludnie.pb")

# %%
