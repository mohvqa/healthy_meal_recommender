import torch
import torch.nn as nn

class HybridRecommender(nn.Module):
    def __init__(self, n_users, n_meals, user_feat_dim, meal_feat_dim, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        d = hyperparams["embedding_dim"]

        # embeddings & biases
        self.user_emb = nn.Embedding(n_users, d)
        self.meal_emb = nn.Embedding(n_meals, d)
        self.user_bias  = nn.Embedding(n_users, 1)
        self.meal_bias  = nn.Embedding(n_meals, 1)
        self.user_emb_dropout = nn.Dropout(p=hyperparams["emb_dropout"])
        self.meal_emb_dropout = nn.Dropout(p=hyperparams["emb_dropout"])
        self.user_feat_proj = nn.Linear(user_feat_dim, d)
        self.meal_feat_proj = nn.Linear(meal_feat_dim, d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=hyperparams["nhead"],
            dim_feedforward=hyperparams["ff_dim"],
            dropout=hyperparams["dropout_prob"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=hyperparams["n_layers"])

        # final projection to score
        self.out_proj = nn.Linear(d, 1)

    def forward(self, user_ids, meal_ids, user_feats, meal_feats, cold_start_mask=None):
        if cold_start_mask is None:
            cold_start_mask = torch.zeros_like(user_ids, dtype=torch.bool)

        bsz, device = user_ids.size(0), user_ids.device
        d = self.user_emb.embedding_dim

        # cold → zeros , warm → normal lookup
        u_emb = torch.where(
            cold_start_mask.unsqueeze(1),
            torch.zeros(bsz, d, device=device),
            self.user_emb_dropout(self.user_emb(user_ids))
        )

        m_emb = self.meal_emb_dropout(self.meal_emb(meal_ids))
        u_feat = self.user_feat_proj(user_feats)
        m_feat = self.meal_feat_proj(meal_feats)

        seq = torch.stack([u_emb, m_emb, u_feat, m_feat], dim=1)
        pooled = self.transformer(seq).mean(dim=1)
        score = self.out_proj(pooled).squeeze(1)

        u_bias = torch.where(
            cold_start_mask,
            torch.zeros(bsz, device=device),
            self.user_bias(user_ids).squeeze(1)
        )
        m_bias = self.meal_bias(meal_ids).squeeze(1)

        return score + u_bias + m_bias