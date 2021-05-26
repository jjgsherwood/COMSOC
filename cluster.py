from sklearn.mixture import GaussianMixture
import umap
import copy
import numba
import numpy as np

@numba.njit()
def XOR_dist(a,b):
    XOR = np.logical_xor(a,b).sum()
    AND = np.logical_and(a,b).sum()
    avg = (a.sum() + b.sum()) / 2

    try:
        return XOR / (len(a) - avg + AND)
    except:
        return 0


def data_transform(profile, n_components):
    profile.embedding = None
    try:
        profile.embedding
    except ValueError:
        pass
    else:
        return

    data = profile.ballots.astype(int)
    subdata = copy.deepcopy(data)
    np.random.shuffle(subdata)

    reducer = umap.UMAP(n_components=4, n_neighbors=15, min_dist=0.5, metric=XOR_dist)
    reducer.fit(subdata[:5000])
    profile.embedding = reducer.transform(data)


def GMM(profile, n_clusters):
    profile.labels = None
    try:
        profile.labels
    except ValueError:
        pass
    else:
        return

    profile.labels = GaussianMixture(n_clusters).fit_predict(profile.embedding)


def label_profile(profile, n_clusters=10, n_components=4):
    data_transform(profile, n_components)
    GMM(profile, n_clusters)


if __name__ == '__main__':
    from approval_profile import *

    # profile = Profile("data/poland_warszawa_2018_praga-poludnie.pb")
    profile = Profile_Synthetic(list(range(4000, 1, -200)), list(range(1000, 1, -50)), budget_distribution=uniform, low=500, high=10000, spread_of_approvals=2.5, sdcavpd=0.3, noise=0.02)
    # profile = Profile_Synthetic(list(range(1100, 100, -10)), list(range(250, 10, -10)), budget_distribution=uniform, low=500, high=10000)

    label_profile(profile)
    print(profile.labels.shape)
