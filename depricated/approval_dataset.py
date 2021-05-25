# %%
import torch

from torch.utils.data import DataLoader, Dataset

class ApprovalDataset(Dataset):
    def __init__(self, profile):
        self.__ballots = profile.ballots

    def __len__(self):
        return self.__ballots.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor(self.__ballots[idx])

    @property
    def get_data_dim(self):
        return self.__ballots.shape[1]

  
if __name__ == "__main__":
    from approval_profile import Profile_Synthetic

    profile = Profile_Synthetic()
    dataset = ApprovalDataset(profile)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
