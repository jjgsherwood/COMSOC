# %%
import torch
import torch.nn as nn
import torch.optim as optim

from model import ApprovalVAE
from approval_dataset import ApprovalDataset
from approval_profile import Profile_Synthetic
from vae_utils import ELBO
from torch.utils.data import DataLoader

def train(dataloader, model, criterion, optimizer, device, iters=1):
    model.train()

    for i in range(iters):
        for batch in dataloader:
            batch = batch.to(device)
            print(batch)
            optimizer.zero_grad()

            logits, mu, log_std = model(batch)
            loss = criterion(logits, batch, mu, log_std)
            print(loss.item())
            loss.backward()
            optimizer.step()
            break


def main(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    lr = 0.000001
    shuffle = True
    
    profile = Profile_Synthetic()
    dataset = ApprovalDataset(profile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model = ApprovalVAE(dataset.get_data_dim, 2, hidden_dims=[8]).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ELBO(nn.BCEWithLogitsLoss(reduction='sum'))

    train(dataloader, model, criterion, optimizer, device)

    

if __name__ == "__main__":
    main()
# %%

# %%
