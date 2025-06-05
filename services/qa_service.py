from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document

qa_chain = None

def build_knowledge_base():
    global qa_chain

    # ✅ Spring 없이 동작하는 mock 데이터
    docs = [
        Document(
            page_content=(
                "식당 이름: 맛있는집\n"
                "주소: 서울시 강남구 맛길 123\n"
                "평점: 4.8\n"
                "영업시간: 10:00 ~ 21:00\n"
                "[김철수] 음식이 정말 맛있어요!\n"
                "[박영희] 친절한 서비스에 감동했어요!"
            )
        ),
        Document(
            page_content=(
                "식당 이름: 피자타운\n"
                "주소: 서울시 마포구 피자길 456\n"
                "평점: 4.5\n"
                "영업시간: 11:00 ~ 23:00\n"
                "[이민수] 도우가 쫄깃하고 맛있어요.\n"
                "[최지우] 분위기도 좋아요!"
            )
        )
    ]

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=store.as_retriever()
    )

def ask(question: str) -> str:
    if qa_chain is None:
        return "❌ QA 시스템이 초기화되지 않았습니다. 먼저 /api/init에 요청하세요."
    return qa_chain.run(question)
