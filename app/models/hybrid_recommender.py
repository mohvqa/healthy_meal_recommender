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

    def forward(self, user_ids, meal_ids, user_feats, meal_feats):
        u = self.user_emb_dropout(self.user_emb(user_ids))
        m = self.meal_emb_dropout(self.meal_emb(meal_ids))

        # Project the dense feature vectors to the same dimension d
        u_feat = self.user_feat_proj(user_feats)
        m_feat = self.meal_feat_proj(meal_feats)

        # Stack into a 4-token sequence
        seq = torch.stack([u, m, u_feat, m_feat], dim=1)

        # Transformer forward
        mask = None  # every token to attend to every other token
        x = self.transformer(seq, src_key_padding_mask=mask)

        # Global mean-pool
        pooled = x.mean(dim=1)

        raw_output = self.out_proj(pooled).squeeze(1)

        # Add biases
        u_bias = self.user_bias(user_ids).squeeze(1)
        m_bias = self.meal_bias(meal_ids).squeeze(1)
        return raw_output + u_bias + m_bias