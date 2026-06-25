"""사용자 확인/거절 신호 학습.

확인(example)과 거절(negative) 요청을 벡터 DB에 저장해 다음 매칭 정확도를 높인다.
"""
import uuid

from qdrant_client.models import Document, PointStruct

import config
from clients import qdrant
from workflows import list_workflows, search_workflows


def _save_signal(workflow_id: str, query: str, kind: str):
    """요청 문장을 해당 워크플로우의 학습 포인트(example/negative)로 저장한다."""
    qdrant.upsert(
        collection_name=config.WORKFLOW_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=Document(text=query, model=config.EMBED_MODEL),
                payload={"workflow_id": workflow_id, "text": query, "kind": kind},
            )
        ],
    )


def save_confirmed(workflow_id: str, query: str):
    """사용자가 확인한 (요청 → 워크플로우) 매핑을 example 포인트로 저장한다.

    저장된 요청 문장이 임베딩되어, 다음에 비슷한 요청이 오면 이 워크플로우가
    더 잘 매칭된다(매칭 정확도 누적 향상).
    같은 워크플로우에 거의 똑같은 요청이 이미 있으면 중복 저장을 생략한다.
    """
    if not any(w["workflow_id"] == workflow_id for w in list_workflows()):
        return
    # 중복 제거: 같은 워크플로우에 이미 매우 유사한 학습 요청이 있으면 건너뜀
    for h in search_workflows(query, limit=5):
        if (h["workflow_id"] == workflow_id and h["kind"] == "example"
                and h["score"] >= config.DEDUP_THRESHOLD):
            return
    _save_signal(workflow_id, query, "example")


def save_rejected(workflow_id: str, query: str):
    """사용자가 거절/교정한 (요청 ↛ 워크플로우) 매핑을 negative 포인트로 저장한다.

    다음에 비슷한 요청이 오면 이 워크플로우의 매칭 점수가 낮아진다(오매칭 감소).
    """
    if not workflow_id:
        return
    _save_signal(workflow_id, query, "negative")
