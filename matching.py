"""요청 → 워크플로우 매칭 (벡터 검색으로 후보를 좁히고 LLM이 최종 확정).

매칭 대상은 system 구분 워크플로우로 한정한다(일반 워크플로우는 제외).
과거 확인(positive)은 점수를 높이고, 과거 거절(negative)은 점수를 낮춘다.
"""
import json

import config
from clients import llm
from workflows import list_workflows, search_workflows


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
    definitions = {w["workflow_id"]: w for w in list_workflows(category="system")}
    if not definitions:
        return {"matched": False, "workflow": None, "score": 0.0,
                "reason": "매칭 가능한 system 워크플로우가 없습니다.", "candidates": []}

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
