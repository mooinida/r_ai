from pydantic import BaseModel
from typing import List

# 식당 정보를 정의하는 모델 (FastAPI + 타입 검증용)
class Restaurant(BaseModel):
    placeId: str
    name: str
    address: str
    rating: float
    latitude: float
    longitude: float
    openingHours: List[str] = []  # 영업시간 리스트