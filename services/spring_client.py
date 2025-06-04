import requests

# Spring 서버 주소 및 JWT 인증
SPRING_SERVER = "http://localhost:8000"
JWT_TOKEN="" # 실제 토큰 입력

headers={"Authorization":f"Bearer {JWT_TOKEN}"}
#headers http 요청에 포함할 헤어 정보를 담은 파이썬 딕셔너리
#authorization 헤더의 키 : 인증 정보를 담는 표준 키
#bearer 방식의 토큰 인증 문자열

#식당 목록 가져오기
def fetch_restaurants():
    url=f"{SPRING_SERVER}/api/restaurants"
    return requests.get(url,headers=headers).json()

#특정 식당의 리뷰 가져오기
def fetch_reviews(place_id):
    url=f"{SPRING_SERVER}/api/restaurants/{place_id}/reviews"
    return requests.get(url,headers=headers).json()