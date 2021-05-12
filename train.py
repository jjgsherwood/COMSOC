# %%
import torch
import torch.nn as nn
import torch.optim as optim

from model import ApprovalVAE
from approval_dataset import ApprovalDataset
from approval_profile import Profile_Synthetic
from vae_utils import ELBO
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(dataloader, model, criterion, optimizer, device, epochs=1, verbatim=False, iter_interval=100):
    model.train()

    if verbatim:
        print(model)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, mu, log_std = model(batch)
            loss = criterion(logits, batch, mu, log_std)
            loss.backward()
            optimizer.step()

            if (verbatim and not i % iter_interval) or (not i and not epoch):
                print(f"Epoch {epoch}, iter {i} - Loss: {loss.item()}")

        print(f"Epoch {epoch} - Loss: {loss.item()}")


def main(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    lr = 0.00002
    shuffle = True
    
    profile = Profile_Synthetic()
    dataset = ApprovalDataset(profile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model = ApprovalVAE(dataset.get_data_dim, 64, hidden_dims=[8]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ELBO(nn.BCEWithLogitsLoss(reduction='sum'))

    train(dataloader, model, criterion, optimizer, device, epochs=10)

    
if __name__ == "__main__":
    main()
# %%

# %%
