from pydantic import BaseModel

# 식당 리뷰 데이터를 표현하는 모델
class Review(BaseModel):
    author: str
    rating: float
    text: str