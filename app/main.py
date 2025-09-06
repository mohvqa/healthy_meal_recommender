from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.utils.recommend import recommend_hybrid_retrieval
from app.utils.similar_meals import get_top_k_similar_meals
from app.utils.top_bias_meals import get_top_k_bias_meals

app = FastAPI()

class RecItem(BaseModel):
    id: int
    score: float
    model_config = {"json_schema_extra": {"example": {"id": 123, "score": 0.987}}}

class SimilarOut(BaseModel):
    similar: List[int]

class BiasOut(BaseModel):
    meals: List[int]

class RecRequest(BaseModel):
    user_id: int
    k: int = 10

@app.post("/recommend", response_model=List[RecItem])
def api_recommend(req: RecRequest):
    try:
        return recommend_hybrid_retrieval(req.user_id, req.k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/similar/{meal_id}", response_model=SimilarOut)
def api_similar_meals(meal_id: int, k: int = 10):
    try:
        return get_top_k_similar_meals(meal_id, k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/top-bias", response_model=BiasOut)
def api_top_bias_meals(k: int = 10):
    try:
        return get_top_k_bias_meals(k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))