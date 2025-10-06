from typing import List
from pydantic import BaseModel


class PredictResponse(BaseModel):
    boxes: List[float] = []  # list of [x1, y1, x2, y2]
    labels: List[int] = []
    # meta: dict = {}