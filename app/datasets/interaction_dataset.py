import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    def __init__(self, user_ids, meal_ids, labels):
        self.user_ids = user_ids
        self.meal_ids = meal_ids
        self.labels = labels

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(self.meal_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

def interaction_collate_fn(batch):
    user_ids, meal_ids, labels = zip(*batch)
    return (
        torch.stack(user_ids),
        torch.stack(meal_ids),
        torch.stack(labels),
    )
