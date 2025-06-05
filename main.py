from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.qa_service import build_knowledge_base, ask

app = FastAPI()

# CORS 설정 (React 연결 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시엔 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# ✅ QA 초기화는 따로 호출하게 만든다
@app.get("/api/init")
def initialize_qa():
    build_knowledge_base()
    return {"message": "QA knowledge base initialized."}

@app.post("/api/ask")
def ask_route(q: Question):
    answer = ask(q.question)
    return {"answer": answer}
