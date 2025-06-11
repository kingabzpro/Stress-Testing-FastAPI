import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

from .ml_model import ml_model
from .pydantic_models import (
    PredictionRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the model
    _ = ml_model.get_feature_info()
    yield


app = FastAPI(
    title="California Housing Price Prediction API",
    version="1.0.0",
    description="API for predicting California housing prices using Random Forest model",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Service is operational"}


@app.get("/model-info")
async def model_info():
    """Get information about the ML model"""
    try:
        feature_info = await asyncio.to_thread(ml_model.get_feature_info)
        return {
            "model_type": "Random Forest Regressor",
            "dataset": "California Housing Dataset",
            "features": feature_info,
        }
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error retrieving model information"
        )


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make house price prediction"""
    if len(request.features) != 8:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 8 features, got {len(request.features)}",
        )
    try:
        prediction = ml_model.predict(request.features)
        return {
            "prediction": float(prediction),
            "status": "success",
            "features_used": request.features,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction error")
