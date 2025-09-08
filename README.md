# 🍽️ Meal Recommender

A **PyTorch + FastAPI** service that suggests meals to users with a **two-stage** (retrieval → reranking) hybrid recommender.  
Built for small-to-medium catalogs (≲50k meals) and updated in near real-time as new ratings arrive.

---

## 🔍 What it does
1. `/recommend`  
   `POST {"user_id": 123, "k": 10}` → returns `k` meals the user is most likely to enjoy.  
   - Filters out meals already rated by the user.  
   - First stage: **RetrievalNet** (two-tower, cosine similarity) produces 50 candidates.  
   - Second stage: **HybridRecommender** (Transformer + embeddings + features) reranks candidates.

2. `/similar/{meal_id}?k=10`  
   Returns the `k` most similar meals via learned meal embeddings.

3. `/top-bias?k=10`  
   Returns the `k` meals with the highest learned bias term (global popularity proxy).

---

## 🧠 Model architecture
| Component | Description |
|-----------|-------------|
| **Embeddings** | 32-dim user & meal embeddings + biases |
| **Features** | Dense user features, meal features, 384-dim meal-description BERT vector |
| **Transformer** | 2-layer encoder, 2 heads, 32-dim feed-forward, residual + dropout |
| **Scoring** | Mean-pool transformer output → linear → logits + biases |
| **Retrieval** | Shared embeddings → small MLP tower → cosine similarity w/ learnable temperature |

---

## 🚀 Quick start (local)
```bash
# 1. Clone
git clone https://github.com/mohvqa/healthy_meal_recommender.git
cd healthy_meal_recommender

# 2. One-command setup (venv + deps)
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# 3. Run server
chmod +x scripts/run_server.sh
./scripts/run_server.sh
# → Swagger at http://localhost:8002/docs