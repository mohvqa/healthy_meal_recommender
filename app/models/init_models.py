import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from app.models.hybrid_recommender import HybridRecommender
from app.models.retrieval_net import RetrievalNet
from app.datasets.meal_rating_dataset import MealRatingDataset
from app.datasets.interaction_dataset import InteractionDataset
from app.datasets.meal_rating_dataset import meal_collate_fn
from app.datasets.interaction_dataset import interaction_collate_fn
from torch.utils.data import DataLoader
from app.config import settings

DATA_DIR = settings.data_dir

# Load core CSVs
users   = pd.read_csv(DATA_DIR / "users.csv")
meals   = pd.read_csv(DATA_DIR / "meals.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")

# Load artifacts
artifact     = joblib.load(DATA_DIR / "artifacts_hybrid_recommender.pkl")
user_preproc = artifact["user_preprocessor"]
meal_preproc = artifact["meal_preprocessor"]
uid_to_idx   = artifact["uid_to_idx"]
mid_to_idx   = artifact["mid_to_idx"]
idx_to_mid   = {v: k for k, v in mid_to_idx.items()}
idx_to_uid   = {v: k for k, v in uid_to_idx.items()}

# Feature arrays: generate once (from preprocessors) and cache to .npy
UF_NPY = DATA_DIR / "user_features.npy"
MF_NPY = DATA_DIR / "meal_features.npy"

if UF_NPY.exists() and MF_NPY.exists():
    user_features = np.load(UF_NPY)
    meal_features = np.load(MF_NPY)
else:
    # Transform using saved preprocessors from the notebook
    USER_COLS = user_preproc.feature_names_in_
    MEAL_COLS = meal_preproc.feature_names_in_

    # users/meals are keyed by "id" in the notebook
    users_idxed = users.set_index("id")
    meals_idxed = meals.set_index("id")

    user_features = user_preproc.transform(users_idxed[USER_COLS]).astype("float32")
    meal_features = meal_preproc.transform(meals_idxed[MEAL_COLS]).astype("float32")

    np.save(UF_NPY, user_features)
    np.save(MF_NPY, meal_features)

DESC_NPY_FILE = DATA_DIR / "meal_desc_384.npy"
desc_vecs = np.load(DESC_NPY_FILE).astype("float32")

# Build & load models
device = torch.device(settings.device)

hyperparams = {
    "embedding_dim": settings.embedding_dim,
    "emb_dropout": settings.emb_dropout,
    "nhead": settings.nhead,
    "ff_dim": settings.ff_dim,
    "dropout_prob": settings.dropout_prob,
    "n_layers": settings.n_layers,
}


base_model = HybridRecommender(
    n_users=len(uid_to_idx),
    n_meals=len(mid_to_idx),
    user_feat_dim=user_features.shape[1],
    meal_feat_dim=meal_features.shape[1] + desc_vecs.shape[1],
    hyperparams=hyperparams,
)
base_model.load_state_dict(torch.load(DATA_DIR / "hybrid_recommender.pt", map_location=device))
base_model.eval()

retrieval_model = RetrievalNet(base_model)
retrieval_model.load_state_dict(torch.load(DATA_DIR / "retrieval.pt", map_location=device))
retrieval_model.eval()

# Public helpers
def build_meal_dataset(df: pd.DataFrame):
    return MealRatingDataset(df, uid_to_idx, mid_to_idx, user_features, meal_features, desc_vecs)

def build_meal_loader(df: pd.DataFrame, batch_size=512, shuffle=True):
    return DataLoader(build_meal_dataset(df), batch_size=batch_size, shuffle=shuffle, collate_fn=meal_collate_fn)

def build_interaction_dataset(user_ids, meal_ids, labels):
    return InteractionDataset(user_ids, meal_ids, labels)

def build_interaction_loader(user_ids, meal_ids, labels, batch_size=512, shuffle=False):
    return DataLoader(build_interaction_dataset(user_ids, meal_ids, labels),
                      batch_size=batch_size, shuffle=shuffle, collate_fn=interaction_collate_fn)