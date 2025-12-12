from pydantic import BaseModel

class TextsRequest(BaseModel):
    texts: list[str]

class PredictionResult(BaseModel):
    text: str
    predicted_class: int

class PredictionResponse(BaseModel):
    batch_size: int
    results: list[PredictionResult]