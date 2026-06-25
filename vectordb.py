"""벡터 DB(Qdrant) 검색·저장 및 문서 학습 관련 로직."""
import uuid

from pypdf import PdfReader
from qdrant_client.models import Document, PointStruct

import config
from clients import qdrant


# ── 컬렉션 현황/관리 ───────────────────────────────────
def count_points() -> int:
    """저장된 항목 수를 반환(실패 시 0)."""
    try:
        return qdrant.get_collection(config.COLLECTION).points_count
    except Exception:
        return 0


def reset_collection():
    """컬렉션을 비우고 새로 만든다."""
    qdrant.delete_collection(config.COLLECTION)
    qdrant.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=config.VECTORS_CONFIG,
    )


# ── 검색 ───────────────────────────────────────────────
def _payload_to_text(payload: dict) -> str:
    # Q&A 형태면 Q/A로, 문서 조각이면 본문으로 변환
    if "answer" in payload:
        return f"Q: {payload['question']}\nA: {payload['answer']}"
    return payload.get("text", "")


def search_similar(query: str) -> list[dict]:
    try:
        results = qdrant.query_points(
            collection_name=config.COLLECTION,
            query=Document(text=query, model=config.EMBED_MODEL),
            limit=config.TOP_K,
        ).points
        return [
            {
                "id": r.id,
                "text": _payload_to_text(r.payload),
                "feedback": r.payload.get("feedback", ""),
                "source": r.payload.get("source"),
                "score": r.score,
            }
            for r in results if r.score > config.SCORE_THRESHOLD
        ]
    except Exception:
        return []


# ── Q&A 저장 ───────────────────────────────────────────
def find_existing_qa(question: str) -> str | None:
    """질문과 매우 유사한 기존 Q&A가 있으면 그 point id를 반환(없으면 None)."""
    try:
        results = qdrant.query_points(
            collection_name=config.COLLECTION,
            query=Document(text=question, model=config.EMBED_MODEL),
            limit=1,
        ).points
        if results and results[0].score >= config.DEDUP_THRESHOLD and "answer" in results[0].payload:
            return results[0].id
    except Exception:
        pass
    return None


def save_to_db(question: str, answer: str, feedback: str = "", point_id: str | None = None):
    payload = {"question": question, "answer": answer}
    if feedback:
        payload["feedback"] = feedback  # 어떤 피드백으로 개선됐는지 기록
    qdrant.upsert(
        collection_name=config.COLLECTION,
        points=[
            PointStruct(
                # point_id가 있으면 기존 항목을 덮어씀(누적 개선), 없으면 새로 추가
                id=point_id or str(uuid.uuid4()),
                vector=Document(text=question, model=config.EMBED_MODEL),
                payload=payload,
            )
        ],
    )


# ── 문서 학습(저장) ────────────────────────────────────
def read_uploaded_file(uploaded) -> str:
    """업로드된 파일에서 텍스트를 추출한다 (pdf는 페이지별로 추출)."""
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """긴 문서를 검색하기 좋은 크기로 잘게 나눈다 (overlap으로 문맥 끊김 완화)."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def save_document(text: str, source: str) -> int:
    """문서를 chunk 단위로 임베딩해서 벡터 DB에 저장한다. 저장한 chunk 수 반환."""
    chunks = chunk_text(text)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=Document(text=c, model=config.EMBED_MODEL),
            payload={"text": c, "source": source},
        )
        for c in chunks
    ]
    if points:
        qdrant.upsert(collection_name=config.COLLECTION, points=points)
    return len(points)
