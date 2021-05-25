# %%
import torch
import torch.nn as nn
import torch.optim as optim

from model import ApprovalVAE
from approval_dataset import ApprovalDataset
from approval_profile import Profile_Synthetic
from vae_utils import ELBO, visualize
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(dataloader, model, criterion, optimizer, device, epochs=1, verbatim=False, iter_interval=100, save=False, path="models/model.pth", vis=False):
    model.train()

    if verbatim:
        print(model)

    for epoch in range(epochs):
        running_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, mu, log_std = model(batch)
            loss = criterion(logits, batch, mu, log_std)
            loss.backward()
            optimizer.step()

            if verbatim and not i % iter_interval:
                print(f"Epoch {epoch}, iter {i} - Loss: {loss.item()}")
                if vis:
                    visualize(model, batch)
            running_loss += loss.item()

        print(f"Epoch {epoch} - Loss: {running_loss / i}")

    if save:
        model.save(path)


def main(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    lr = 0.0001
    shuffle = True
    
    profile = Profile_Synthetic()
    dataset = ApprovalDataset(profile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model = ApprovalVAE(dataset.get_data_dim, 4, hidden_dims=[8,6]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ELBO(nn.BCEWithLogitsLoss(reduction='sum'))

    train(dataloader, model, criterion, optimizer, device, epochs=10, save=False, verbatim=True, vis=True)

    visualize(model, torch.Tensor(profile.ballots[:40]))
    

    
if __name__ == "__main__":
    main()
# %%

# %%
