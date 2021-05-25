# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_utils import reparamaterize


class ApprovalVAE(nn.Module):
    def __init__(self, input_dim=12, latent_dim=2, hidden_dims=[75,50,25]):
        super(ApprovalVAE, self).__init__()

        self.encoder = LinearMLP(input_dim, latent_dim*2, hidden_dims=hidden_dims)
        self.decoder = LinearMLP(latent_dim, input_dim, hidden_dims=hidden_dims[::-1])
        self.act_fn = torch.sigmoid


    def forward(self, x):
        mu, log_std = torch.chunk(self.encoder(x), 2, 1)
        samples = reparamaterize(mu, log_std).to(x.device)
        logits = self.decoder(samples)
        return logits, mu, log_std


    def reconstruct(self, x):
        with torch.no_grad():
            logits, *_ = self(x)
        return torch.sigmoid(logits)

    
    def save(self, path):
        torch.save(self, path)

    
    @staticmethod
    def load(path):
        model = torch.load(path)
        model.eval()
        return model


class LinearMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[75,50,25]):
        super(LinearMLP, self).__init__()

        dims = [input_dim] + hidden_dims
        self.act_fn = nn.ReLU()
        layers = []

        for in_features, out_features in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(self.act_fn)
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    from approval_profile import Profile_Synthetic
    from approval_dataset import ApprovalDataset
    from torch.utils.data import DataLoader
    from vae_utils import ELBO

    profile = Profile_Synthetic()
    dataset = ApprovalDataset(profile)
    dataloader = DataLoader(dataset, batch_size=2)

    model = ApprovalVAE(12, 2)

    criterion = ELBO(nn.BCEWithLogitsLoss(reduction='sum'))

    for batch in dataloader:
        logits, mu, log_std = model(batch)
        loss = criterion(logits, batch, mu, log_std)
        print(loss.item())
        break
# %%
