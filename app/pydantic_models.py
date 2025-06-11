from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude",
        min_length=8,
        max_length=8,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"features": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]}
            ]
        }
    }
