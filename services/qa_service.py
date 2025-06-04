import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
from services.spring_client import fetch_restaurants, fetch_reviews

qa_chain = None

def build_knowledge_base():
    global qa_chain

    restaurants = fetch_restaurants()
    docs = []

    for r in restaurants:
        reviews = fetch_reviews(r["placeId"])
        text = f"식당 이름: {r['name']}\n주소: {r['address']}\n평점: {r['rating']}\n"
        text += f"영업시간: {'; '.join(r.get('openingHours', []))}\n"
        text += "\n".join([f"[{v['author']}] {v['text']}" for v in reviews])
        docs.append(Document(page_content=text))

    # OpenAIEmbeddings와 OpenAI는 자동으로 os.environ['OPENAI_API_KEY']를 참조
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=store.as_retriever()
    )

def ask(question: str) -> str:
    if qa_chain is None:
        return "❌ QA 시스템이 초기화되지 않았습니다. 먼저 build_knowledge_base()를 호출하세요."
    return qa_chain.run(question)
