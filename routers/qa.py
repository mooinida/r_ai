from fastapi import APIRouter, Query
from services.qa_service import ask

router = APIRouter(tags=["QA"])

# 사용자가 자연어로 질문하면 LangChain이 답변
@router.get("/")
def qa_endpoint(q: str = Query(..., description="예: 강남역 맛집 알려줘")):
    return {"answer": ask(q)}