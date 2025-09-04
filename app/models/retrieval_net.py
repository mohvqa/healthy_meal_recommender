import torch
import torch.nn as nn
import torch.nn.functional as F
from app.models.hybrid_recommender import HybridRecommender

class RetrievalNet(nn.Module):
    # Two-tower retrieval model that re-uses the embeddings already trained by HybridRecommender.
    def __init__(self, base: HybridRecommender):
        super().__init__()
        self.user_emb = base.user_emb
        self.meal_emb = base.meal_emb
        
        # small MLP on top of each tower
        d = base.user_emb.embedding_dim
        self.user_tower = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.meal_tower = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, user_ids, meal_ids):
        u = self.user_tower(self.user_emb(user_ids))
        m = self.meal_tower(self.meal_emb(meal_ids))
        return F.cosine_similarity(u, m, dim=-1) * self.temperature