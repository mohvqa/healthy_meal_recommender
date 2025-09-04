from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.utils.recommend import recommend_hybrid_retrieval
from app.utils.similar_meals import get_top_k_similar_meals
from app.utils.top_bias_meals import get_top_k_bias_meals

app = FastAPI()

class RecRequest(BaseModel):
    user_id: int
    k: int = 10

@app.post("/recommend")
def api_recommend(req: RecRequest):
    try:
        return recommend_hybrid_retrieval(req.user_id, req.k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/similar/{meal_id}")
def api_similar_meals(meal_id: int, k: int = 10):
    try:
        return get_top_k_similar_meals(meal_id, k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/top-bias")
def api_top_bias_meals(k: int = 10):
    try:
        return get_top_k_bias_meals(k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))