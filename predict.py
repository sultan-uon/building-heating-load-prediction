"""Serve the saved Heating Load pipeline with FastAPI."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Building(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    relative_compactness: float = Field(ge=0.62, le=0.98)
    surface_area: float = Field(ge=514.5, le=808.5)
    wall_area: float = Field(ge=245, le=416.5)
    roof_area: float = Field(ge=110.25, le=220.5)
    overall_height: float = Field(ge=3.5, le=7)
    orientation: int = Field(ge=2, le=5)
    glazing_area: float = Field(ge=0, le=0.4)
    glazing_area_distribution: int = Field(ge=0, le=5)

    @model_validator(mode="after")
    def consistent_glazing(self):
        if (self.glazing_area == 0) != (self.glazing_area_distribution == 0):
            raise ValueError("Zero glazing requires distribution 0; positive glazing requires 1–5.")
        return self


@asynccontextmanager
async def lifespan(app):
    path = Path(__file__).resolve().parent / "models/heating_model.joblib"
    try:
        app.state.model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError("Cannot load heating_model.joblib; run python train.py first.") from exc
    yield


app = FastAPI(title="Heating Load Prediction", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Keep nonfinite input values and exception objects out of JSON errors.
    errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(building: Building):
    frame = pd.DataFrame([building.model_dump()])
    return {"heating_load": float(app.state.model.predict(frame)[0])}
