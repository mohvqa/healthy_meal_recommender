import json
from app.models.init_models import idx_to_mid, base_model

def get_top_k_bias_meals(k: int = 10):
    meal_bias = base_model.meal_bias.weight.squeeze().detach().cpu()
    idxs = meal_bias.argsort(descending=True)[:k]

    meal_ids = [idx_to_mid[idx.item()] for idx in idxs]
    return {"meals": meal_ids}