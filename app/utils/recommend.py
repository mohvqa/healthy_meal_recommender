import pandas as pd
import numpy as np
import torch, json
from app.models.init_models import build_meal_loader
from app.models.init_models import ( base_model, retrieval_model, uid_to_idx, mid_to_idx, idx_to_mid, ratings, )
from app.config import settings

@torch.no_grad()
def recommend_hybrid_retrieval(user_id: int, k: int = 10, num_candidates: int = 50, device=None):
    if device is None:
        device = settings.device

    if user_id not in uid_to_idx:
        raise Exception(f"User {user_id} not found in the dataset.")

    user_idx = uid_to_idx[user_id]
    user_tensor = torch.tensor([user_idx], device=device)

    # Stage 1: Retrieval (get initial candidates)
    all_meals_tensor = torch.arange(len(mid_to_idx), device=device)
    retrieval_scores = torch.sigmoid(retrieval_model(user_tensor, all_meals_tensor))

    # Get top_candidates meal indices from the retrieval model
    top_candidate_indices = torch.topk(retrieval_scores, k=num_candidates).indices.tolist()
    candidate_meal_ids = [idx_to_mid[idx] for idx in top_candidate_indices]

    # Filter out meals the user has already rated
    rated_meal_ids = set(ratings.loc[ratings["user_id"] == user_id, "meal_id"])
    candidate_meal_ids = [meal_id for meal_id in candidate_meal_ids if meal_id not in rated_meal_ids]

    if not candidate_meal_ids:
        raise Exception(f"No new candidate meals found for user {user_id}.")

    # Stage 2: Reranking (use Hybrid model on candidates)
    cand_ratings_df = pd.DataFrame({
        'user_id': user_id,
        'meal_id': candidate_meal_ids,
        'rating': 0
    })

    cand_loader = build_meal_loader(cand_ratings_df, batch_size=512)

    all_rerank_scores = []

    for batch in cand_loader:
        u_ids_batch = batch["user_idx"].to(device)
        m_ids_batch = batch["meal_idx"].to(device)
        u_feats_batch = batch["user_feat"].to(device)
        m_feats_batch = batch["meal_feat"].to(device)

        scores_batch = base_model(u_ids_batch, m_ids_batch, u_feats_batch, m_feats_batch).cpu().numpy()
        all_rerank_scores.extend(scores_batch)

    rerank_scores = np.array(all_rerank_scores)
    top_k_indices = np.argpartition(-rerank_scores, k)[:k]

    final_recommendation_ids = [candidate_meal_ids[i] for i in top_k_indices]
    final_recommendation_scores = rerank_scores[top_k_indices]

    recommended_meals_df = pd.DataFrame({
        'id': final_recommendation_ids,
        'score': final_recommendation_scores
    })

    return json.loads(recommended_meals_df.sort_values("score", ascending=False).reset_index(drop=True).to_json(orient="records"))