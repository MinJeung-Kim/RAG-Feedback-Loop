"""워크플로우 저장소: 생성·조회 및 벡터 검색 (Qdrant 접근 계층).

벡터 DB에는 세 종류의 포인트가 섞여 있다 (payload["kind"]로 구분):
  - "definition" : 워크플로우 정의 (이름+설명) — 목록/후보에 노출
  - "example"    : 사용자가 확인한 과거 요청 — 매칭 점수를 높이는 데 사용(목록엔 미노출)
  - "negative"   : 사용자가 거절/교정한 과거 요청 — 매칭 점수를 낮추는 데 사용(목록엔 미노출)
"""
import uuid

from qdrant_client.models import Document, PointStruct

import config
from clients import qdrant


def _new_id() -> str:
    """워크플로우 고유 id 발급 (예: wf_3f9a1c2b7d40)."""
    return "wf_" + uuid.uuid4().hex[:12]


def create_workflow(name: str, description: str, category: str = "general") -> dict:
    """이름·설명·구분(system/general)으로 워크플로우를 만들고 고유 id를 발급해 저장한다."""
    wf = {
        "workflow_id": _new_id(),
        "name": name.strip(),
        "description": description.strip(),
        "category": category,  # "system" | "general"
    }
    qdrant.upsert(
        collection_name=config.WORKFLOW_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                # 이름+설명을 함께 임베딩해 의미 매칭 품질을 높임
                vector=Document(text=f"{wf['name']}\n{wf['description']}", model=config.EMBED_MODEL),
                payload={**wf, "kind": "definition"},
            )
        ],
    )
    return wf


def list_workflows(category: str | None = None) -> list[dict]:
    """등록된 워크플로우 정의 목록을 반환한다 (학습용 example/negative 포인트는 제외).

    category 를 주면 해당 구분("system"/"general")만 필터링한다.
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=config.WORKFLOW_COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return []
    defs = [p.payload for p in points if p.payload.get("kind", "definition") == "definition"]
    if category:
        defs = [d for d in defs if d.get("category", "general") == category]
    return defs


def count_workflows() -> int:
    return len(list_workflows())


def search_workflows(query: str, limit: int = 10) -> list[dict]:
    """요청과 유사한 포인트(정의/확인/거절 모두)를 벡터 검색으로 가져온다."""
    try:
        results = qdrant.query_points(
            collection_name=config.WORKFLOW_COLLECTION,
            query=Document(text=query, model=config.EMBED_MODEL),
            limit=limit,
        ).points
    except Exception:
        return []
    return [
        {
            "workflow_id": r.payload.get("workflow_id"),
            "kind": r.payload.get("kind", "definition"),
            "score": r.score,
        }
        for r in results if r.payload.get("workflow_id")
    ]
