import pickle
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from collections import Counter
import umap
import networkx as nx
from itertools import permutations
import hdbscan
import numpy as np
from mechanism_solver import MechanismAStarSolver, MechanismDynamicSolver
from approval_profile import *

def 

if __name__ == '__main__':
    # profile = Profile_Synthetic(list(range(60000, 100, -3000)), list(range(10000, 1, -1000)), budget_distribution=uniform, low=500, high=10000)

    profile = Profile_Synthetic.load('gen_data//55000_20_liniear.pb')
    data = profile.ballots
    old_clusters = profile.clusters
    old_labels = [j for i,cluster in enumerate(old_clusters) for j in len(cluster)*[i]]

    reducer = umap.UMAP(n_components=4, n_neighbors=100, metric='manhattan')
    reducer.fit(data)
    embedding = reducer.transform(data)

    n_clusters = len(old_clusters)
    k = GaussianMixture(n_clusters).fit(embedding)
    k.labels_ = k.predict(embedding)

    mechanism = Mechanism(profile)
    projects = mechanism.solve('max_approval')