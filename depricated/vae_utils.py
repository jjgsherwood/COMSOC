# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

class ELBO():
    def __init__(self, criterion):
        self.criterion = criterion


    def __call__(self, logits, target, mu, log_std):
        return self.criterion(logits, target) + ELBO.KLD(mu, log_std)


    @staticmethod
    def KLD(mu, log_std):
        return 0.5*torch.sum(torch.exp(2*log_std) + mu**2 - 2*log_std - 1)


def reparamaterize(mu, log_std):
    return torch.exp(log_std)*torch.randn_like(mu) + mu


def visualize(model, ballots):
    reconstruction = model.reconstruct(ballots)
    img = np.array(torch.cat((ballots, torch.ones((len(ballots), 1)) - 0.5, torch.round(reconstruction)), dim=1))
    plt.imshow(img, interpolation='none')
    plt.show()
    

# %%
