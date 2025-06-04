from fastapi import APIRouter, Path
from services.spring_client import fetch_reviews

router = APIRouter(tags=["Reviews"])

# 특정 식당의 리뷰 조회
@router.get("/{place_id}")
def get_reviews(place_id: str = Path(...)):
    return fetch_reviews(place_id)