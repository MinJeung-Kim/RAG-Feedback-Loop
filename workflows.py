"""사용자가 만든 워크플로우 관리 + 요청을 워크플로우로 매칭하는 로직.

흐름:
  1) create_workflow  : 이름·설명을 받아 고유 id를 발급하고, 임베딩해 벡터 DB에 저장
  2) match_workflow   : 사용자 요청을 임베딩해 가장 유사한 워크플로우를 찾고 LLM이 최종 확정
  3) run_workflow_api : (사용자 확인 후) 확정된 워크플로우 id로 가상 API 호출
  4) save_confirmed   : 사용자가 확인한 (요청 → 워크플로우) 매핑을 저장해 매칭 정확도를 높임

벡터 DB에는 세 종류의 포인트가 섞여 있다 (payload["kind"]로 구분):
  - "definition" : 워크플로우 정의 (이름+설명) — 목록/후보에 노출
  - "example"    : 사용자가 확인한 과거 요청 — 매칭 점수를 높이는 데 사용(목록엔 미노출)
  - "negative"   : 사용자가 거절/교정한 과거 요청 — 매칭 점수를 낮추는 데 사용(목록엔 미노출)
"""
import json
import uuid

from qdrant_client.models import Document, PointStruct

import config
from clients import llm, qdrant


def _new_id() -> str:
    """워크플로우 고유 id 발급 (예: wf_3f9a1c2b7d40)."""
    return "wf_" + uuid.uuid4().hex[:12]


# ── 1) 워크플로우 생성 / 조회 ──────────────────────────
def create_workflow(name: str, description: str) -> dict:
    """이름·설명으로 워크플로우를 만들고 고유 id를 발급해 벡터 DB에 저장한다."""
    wf = {"workflow_id": _new_id(), "name": name.strip(), "description": description.strip()}
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


def list_workflows() -> list[dict]:
    """등록된 워크플로우 정의 목록을 반환한다 (학습용 example 포인트는 제외)."""
    try:
        points, _ = qdrant.scroll(
            collection_name=config.WORKFLOW_COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return []
    return [p.payload for p in points if p.payload.get("kind") != "example"]


def count_workflows() -> int:
    return len(list_workflows())


# ── 2) 요청 → 워크플로우 매칭 (벡터 + LLM 확정) ───────
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


def _parse_json(raw: str) -> dict:
    """LLM 출력에서 JSON 객체만 뽑아 파싱(잡음 제거)."""
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except Exception:
        return {"workflow_id": "", "reason": "JSON 파싱 실패"}


def match_workflow(query: str) -> dict:
    """요청에 가장 알맞은 워크플로우를 벡터+LLM으로 찾는다.

    반환: {matched: bool, workflow: dict|None, score: float,
           reason: str, candidates: list[dict]}
    """
    definitions = {w["workflow_id"]: w for w in list_workflows()}
    if not definitions:
        return {"matched": False, "workflow": None, "score": 0.0,
                "reason": "등록된 워크플로우가 없습니다.", "candidates": []}

    # 정의/확인(positive)/거절(negative) 포인트를 검색해 워크플로우별 최고 유사도로 집계
    pos: dict[str, float] = {}   # 정의 + 확인 학습 → 점수 ↑
    neg: dict[str, float] = {}   # 거절/교정 학습 → 점수 ↓
    for h in search_workflows(query):
        wid = h["workflow_id"]
        if wid not in definitions:
            continue
        bucket = neg if h["kind"] == "negative" else pos
        if h["score"] > bucket.get(wid, -1.0):
            bucket[wid] = h["score"]

    # 과거 거절과 유사하면 페널티를 줘 순위를 낮춤(adj_score 기준 정렬)
    candidates = []
    for wid, sc in pos.items():
        penalty = neg.get(wid, 0.0)
        penalized = penalty >= config.NEG_THRESHOLD
        candidates.append({
            **definitions[wid],
            "score": sc,
            "adj_score": sc - penalty if penalized else sc,
            "penalized": penalized,
        })
    candidates.sort(key=lambda c: c["adj_score"], reverse=True)
    if not candidates:
        return {"matched": False, "workflow": None, "score": 0.0,
                "reason": "유사한 워크플로우가 없습니다.", "candidates": []}

    listing = "\n".join(
        f'- id={c["workflow_id"]} | {c["name"]}: {c["description"]}' for c in candidates
    )
    system = (
        "당신은 사용자 요청에 가장 알맞은 워크플로우를 고르는 분류기입니다.\n"
        "아래 후보 중 요청을 처리하기에 가장 적합한 워크플로우의 id를 고르세요.\n"
        "적합한 것이 없으면 workflow_id 를 빈 문자열로 두세요.\n"
        "설명·머리말 없이 JSON 객체 하나만 출력합니다. 형식:\n"
        '{"workflow_id": "<id 또는 빈 문자열>", "reason": "<짧은 근거>"}\n\n'
        "[후보]\n" + listing
    )
    resp = llm.chat.completions.create(
        model=config.VLLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=200,
    )
    decision = _parse_json(resp.choices[0].message.content)

    wf_id = (decision.get("workflow_id") or "").strip()
    chosen = next((c for c in candidates if c["workflow_id"] == wf_id), None)
    return {
        "matched": chosen is not None,
        "workflow": chosen,
        "score": chosen["score"] if chosen else candidates[0]["score"],
        "reason": decision.get("reason", ""),
        "candidates": candidates,
    }


# ── 2-1) 매칭 실패 시 LLM 직접 답변 (fallback) ─────────
def general_answer(query: str, history: list[dict] | None = None) -> str:
    """어느 워크플로우에도 맞지 않을 때 LLM이 사용자 요청에 직접 답한다.

    history: [{"question": ..., "answer": ...}, ...] 형태의 최근 대화.
    최근 몇 턴을 함께 넣어 단기 기억(맥락)을 제공한다.
    """
    messages = [{"role": "system", "content": "당신은 친절한 도우미입니다. 사용자 질문에 성실하게 답하세요."}]
    for turn in (history or [])[-config.HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})

    resp = llm.chat.completions.create(
        model=config.VLLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content


# ── 3) 확인 후 API 호출 ────────────────────────────────
def run_workflow_api(workflow_id: str, request: str) -> dict:
    """확정된 워크플로우 id로 가상 API를 호출한다 (실행 시뮬레이션)."""
    wf = next((w for w in list_workflows() if w["workflow_id"] == workflow_id), None)
    if not wf:
        return {"status": "error", "message": f"워크플로우 {workflow_id} 를 찾을 수 없습니다."}
    return {
        "status": "executed",
        "endpoint": f"POST /api/workflows/{workflow_id}/run",
        "workflow_id": workflow_id,
        "workflow_name": wf["name"],
        "request": request,
        "message": f"워크플로우 '{wf['name']}'를 실행했습니다.",
    }


# ── 4) 확인/거절 요청 저장 (학습) ──────────────────────
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
