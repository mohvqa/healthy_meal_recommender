import pandas as pd
import numpy as np
import torch
from app.models.init_models import build_meal_loader
from app.models.init_models import ( base_model, meals, user_preproc )
from app.config import settings

def recommend_cold(user_features_dict: dict, k: int = 10):
    """
    Recommend k meals to a brand-new user given raw attributes.
    user_features_dict : dict with keys that your preprocessor expects
    """
    base_model.eval()
    with torch.inference_mode():
        device = next(base_model.parameters()).device

        # 1. preprocess raw attributes -> 1×F vector
        cold_X = user_preproc.transform(pd.DataFrame([user_features_dict])).astype("float32")[0]

        # 2. full meal catalogue (nothing rated)
        cand_df = cand_df = pd.DataFrame({
            "user_id": -1,
            "meal_id": meals["id"].values,
            "rating": 0.0,  # dummy – never used in inference
        })

        # 3. dataset / loader in COLD mode
        cand_loader = build_meal_loader(cand_df, cold=True, cold_user_features=cold_X,)

        # 4. score all candidates
        all_scores = []
        for batch in cand_loader:
            bsz = batch["meal_idx"].size(0)
            cold_mask = torch.ones(bsz, dtype=torch.bool, device=device)

            scores = base_model(
                batch["user_idx"].to(device),   # dummy idx
                batch["meal_idx"].to(device),
                batch["user_feat"].to(device),  # replicated cold_X
                batch["meal_feat"].to(device),
                cold_mask,
            )
            all_scores.extend(scores.cpu().numpy())

        # 5. top-k
        scores = np.array(all_scores)
        top_k = np.argpartition(-scores, k)[:k]
        rec_df = (
            meals[meals["id"].isin(cand_df["meal_id"].values[top_k])]
            [["id", "title"]]
            .assign(score=scores[top_k])
            .sort_values("score", ascending=False)
        )
        return rec_df[["id", "title", "score"]].to_dict(orient="records")