from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
from app.utils.recommend import recommend_hybrid_retrieval
from app.utils.similar_meals import get_top_k_similar_meals
from app.utils.top_bias_meals import get_top_k_bias_meals
from app.utils.cold_user_recommend import recommend_cold

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

class ColdRecRequest(BaseModel):
    Gender: Literal["Male", "Female"]
    Activity_Level: Literal[
        "Sedentary",
        "Lightly Active",
        "Moderately Active",
        "Very Active",
        "Extremely Active",
    ] = Field(alias="Activity Level")
    Ages: int
    Height: float
    Weight: float
    Daily_Calorie_Target: int = Field(alias="Daily Calorie Target")
    Protein: float
    Sugar: float
    Sodium: float
    Calories: float
    Carbohydrates: float
    Fiber: float
    Fat: float
    Acne: int
    Diabetes: int
    Heart_Disease: int = Field(alias="Heart Disease")
    Hypertension: int
    Kidney_Disease: int = Field(alias="Kidney Disease")
    Weight_Gain: int = Field(alias="Weight Gain")
    Weight_Loss: int = Field(alias="Weight Loss")
    k: int = 10

@app.post("/recommend", response_model=List[RecItem])
def api_recommend(req: RecRequest):
    try:
        return recommend_hybrid_retrieval(req.user_id, req.k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@app.post("/recommend-cold", response_model=List[RecItem])
def api_recommend_cold(req: ColdRecRequest):
    try:
        recommended_meals = recommend_cold(req.dict(by_alias=True), req.k)
        return recommended_meals
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