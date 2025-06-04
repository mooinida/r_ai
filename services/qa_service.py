from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document

from services.spring_client import fetch_restaurants, fetch_reviews

# 전역 QA 체인
qa_chain = None

# LangChain 기반 질의응답 시스템 초기화 함수
def build_knowledge_base():
    global qa_chain

    # Spring에서 식당 정보 수집
    restaurants = fetch_restaurants()
    docs = []

    for r in restaurants:
        reviews = fetch_reviews(r["placeId"])  # ✅ key 이름 확인 (placed → placeId로 추정)
        
        # LangChain 문서 형식 구성
        text = f"식당 이름: {r['name']}\n주소: {r['address']}\n평점: {r['rating']}\n"
        text += f"영업시간: {'; '.join(r.get('openingHours', []))}\n"
        text += "\n".join([f"[{v['author']}] {v['text']}" for v in reviews])
        
        docs.append(Document(page_content=text))

    # 벡터 임베딩 및 벡터스토어 구축
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)

    # QA 체인 생성 (FAISS 기반)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=store.as_retriever()
    )

# 질문에 대해 답변 생성
def ask(question: str) -> str:
    if qa_chain is None:
        return "❌ QA 시스템이 초기화되지 않았습니다. 먼저 build_knowledge_base()를 호출하세요."
    return qa_chain.run(question)
