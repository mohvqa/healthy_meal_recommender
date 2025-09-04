import torch, json
from app.models.init_models import retrieval_model, mid_to_idx, idx_to_mid

def get_top_k_similar_meals(meal_id: int, k: int = 10):
    if meal_id not in mid_to_idx:
        raise Exception(f"Meal {meal_id} not found in the dataset.")

    target_idx = mid_to_idx[meal_id]
    emb_target = retrieval_model.meal_emb.weight[target_idx].unsqueeze(0)
    cos_sim = torch.nn.functional.cosine_similarity(
        emb_target, retrieval_model.meal_emb.weight, dim=1
    )

    top_indices = cos_sim.argsort(descending=True)[:k + 1]
    top_meal_ids = [idx_to_mid[i.item()] for i in top_indices]

    return json.dumps({"original": meal_id, "similar": top_meal_ids[1:]})