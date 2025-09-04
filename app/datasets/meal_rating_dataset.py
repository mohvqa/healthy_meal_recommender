import torch
from torch.utils.data import Dataset

class MealRatingDataset(Dataset):
    def __init__(self, df, uid_to_idx, mid_to_idx, user_features, meal_features, desc_vecs):
        self.df = df.reset_index(drop=True)
        self.uid_to_idx = uid_to_idx
        self.mid_to_idx = mid_to_idx
        self.user_features = user_features
        self.meal_features = meal_features
        self.desc_vecs = desc_vecs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u_idx = self.uid_to_idx[row["user_id"]]
        m_idx = self.mid_to_idx[row["meal_id"]]

        user_feat = torch.tensor(self.user_features[u_idx], dtype=torch.float32)
        meal_feat = torch.tensor(self.meal_features[m_idx], dtype=torch.float32)
        desc_feat = torch.tensor(self.desc_vecs[m_idx], dtype=torch.float32)
        combined_meal_feat = torch.cat((meal_feat, desc_feat), dim=-1)

        rating = torch.tensor(row["rating"], dtype=torch.float32)

        return {
            "user_idx": torch.tensor(u_idx, dtype=torch.long),
            "meal_idx": torch.tensor(m_idx, dtype=torch.long),
            "user_feat": user_feat,
            "meal_feat": combined_meal_feat,
            "rating": rating,
        }

def meal_collate_fn(batch):
    return {
        "user_idx": torch.stack([b["user_idx"] for b in batch]),
        "meal_idx": torch.stack([b["meal_idx"] for b in batch]),
        "user_feat": torch.stack([b["user_feat"] for b in batch]),
        "meal_feat": torch.stack([b["meal_feat"] for b in batch]),
        "rating": torch.stack([b["rating"] for b in batch]),
    }