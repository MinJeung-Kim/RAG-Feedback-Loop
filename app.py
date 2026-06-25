import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Document, PointStruct, VectorParams

load_dotenv()

# ── 설정 ──────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("COLLECTION")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))
EMBED_MODEL = os.getenv("EMBED_MODEL")
TOP_K = int(os.getenv("TOP_K"))

VLLM_URL = os.getenv("VLLM_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
VLLM_MODEL = os.getenv("VLLM_MODEL")

SCORE_THRESHOLD = 0.7  # 이 유사도를 넘는 검색 결과만 참고 자료로 사용
TEMPERATURE = 0.7
MAX_TOKENS = 512

VECTORS_CONFIG = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)


# ── 클라이언트 초기화 ──────────────────────────────────
@st.cache_resource
def init_clients():
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
    )
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VECTORS_CONFIG,
        )
    llm = OpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY)
    return qdrant, llm


qdrant, llm = init_clients()

# ── 벡터 DB 검색 ───────────────────────────────────────
def search_similar(query: str) -> list[dict]:
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=Document(text=query, model=EMBED_MODEL),
            limit=TOP_K,
        ).points
        return [
            {"question": r.payload["question"], "answer": r.payload["answer"], "score": r.score}
            for r in results if r.score > SCORE_THRESHOLD
        ]
    except Exception:
        return []

# ── LLM 답변 생성 ──────────────────────────────────────
def generate_answer(query: str, context: list[dict]) -> str:
    if context:
        ctx_text = "\n".join([
            f"Q: {c['question']}\nA: {c['answer']}" for c in context
        ])
        system = f"""당신은 친절한 추천 도우미입니다.
아래 참고 자료를 바탕으로 사용자 질문에 답하세요.
참고 자료가 없으면 일반 지식으로 답하세요.

[참고 자료]
{ctx_text}"""
    else:
        system = "당신은 친절한 추천 도우미입니다. 사용자 질문에 성실하게 답하세요."

    resp = llm.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content

# ── 벡터 DB에 저장 ─────────────────────────────────────
def save_to_db(question: str, answer: str):
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=Document(text=question, model=EMBED_MODEL),
                payload={"question": question, "answer": answer},
            )
        ],
    )

# ── UI ────────────────────────────────────────────────
st.title("벡터 DB 추천 실습")
st.caption("질문하면 LLM이 답변하고, 좋은 답변은 DB에 쌓여서 점점 똑똑해져요.")

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None  # 피드백 대기 중인 Q&A

# ── 사이드바: DB 현황 ──────────────────────────────────
with st.sidebar:
    st.subheader("DB 현황")
    try:
        info = qdrant.get_collection(COLLECTION)
        count = info.points_count
        st.metric("저장된 Q&A 수", count)
    except Exception:
        st.metric("저장된 Q&A 수", 0)

    if st.button("DB 초기화"):
        qdrant.delete_collection(COLLECTION)
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VECTORS_CONFIG,
        )
        st.session_state.history = []
        st.session_state.pending = None
        st.success("초기화 완료!")
        st.rerun()

# ── 대화 히스토리 출력 ────────────────────────────────
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        if item.get("from_db"):
            st.caption(f"DB 참고 (유사도 {item['top_score']:.2f})")

# ── 피드백 UI (답변 직후) ─────────────────────────────
if st.session_state.pending:
    p = st.session_state.pending
    with st.chat_message("user"):
        st.write(p["question"])
    with st.chat_message("assistant"):
        st.write(p["answer"])
        if p.get("from_db"):
            st.caption(f"DB 참고 (유사도 {p['top_score']:.2f})")

    st.markdown("**이 답변이 도움이 됐나요?**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 맞아요, DB에 저장!", use_container_width=True):
            save_to_db(p["question"], p["answer"])
            p["saved"] = True
            st.session_state.history.append(p)
            st.session_state.pending = None
            st.success("DB에 저장됐어요! 다음 유사 질문에 활용됩니다.")
            st.rerun()

    with col2:
        if st.button("👎 아니요, 그냥 넘어갈게요", use_container_width=True):
            p["saved"] = False
            st.session_state.history.append(p)
            st.session_state.pending = None
            st.rerun()

# ── 입력창 ────────────────────────────────────────────
if not st.session_state.pending:
    query = st.chat_input("무엇이든 물어보세요...")
    if query:
        with st.spinner("검색 중..."):
            context = search_similar(query)

        with st.spinner("답변 생성 중..."):
            answer = generate_answer(query, context)

        st.session_state.pending = {
            "question": query,
            "answer": answer,
            "from_db": len(context) > 0,
            "top_score": context[0]["score"] if context else 0.0,
        }
        st.rerun()