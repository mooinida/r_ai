from fastapi import APIRouter
from services.spring_client import fetch_restaurants

router = APIRouter(tags=["Restaurants"])

# Spring에서 식당 리스트 조회 후 반환
@router.get("/")
def get_all_restaurants():
    return fetch_restaurants()
