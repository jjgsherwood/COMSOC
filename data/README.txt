There are three types of ballots: top_choice, k_approval, approval. The filenames can be found in the pickle file as a dictionary with the three types and with values a list containing tuples (file, number of projects, number of voters.

In the metadata a approval length is given for top_choice and k_approval, which is respectifically 1 and k (under the name "max_length")
In the metadata a max budget is given for approval, which is the same as the budget (under the name "max_sum_cost")
The minimum length is given by "min_length" and is alwasy 1

Other important metadata is:
 - "budget"
 - "num_votes"
 - "num_projects"

 