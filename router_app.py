"""의미 기반 워크플로우 라우터 (Streamlit).

1) 워크플로우 생성: 이름·설명 입력 → 고유 id 발급 → 벡터 DB 저장
2) 요청 입력 → 의도 분석으로 가장 알맞은 워크플로우 매칭
3) 사용자 확인(확인/취소) 후에만 해당 워크플로우 id로 API 호출
4) 어느 워크플로우에도 안 맞으면 LLM이 직접 답변(fallback)
실행: streamlit run router_app.py
"""
import streamlit as st

import config
from executor import run_workflow_api
from fallback import general_answer
from feedback import save_confirmed, save_rejected
from matching import match_workflow
from workflows import count_workflows, create_workflow, list_workflows

st.title("의미 기반 워크플로우 라우터")
st.caption("요청 의도를 분석해 알맞은 워크플로우를 찾고, 확인을 거친 뒤에 실행합니다.")

# ── 세션 상태 ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []     # 대화 내역(단기 기억) [{question, answer, kind}]
if "pending" not in st.session_state:
    st.session_state.pending = None   # 확인 대기 중인 매칭 결과
if "result" not in st.session_state:
    st.session_state.result = None    # 마지막 실행 결과 상세(JSON)


# ── 실행 컨트롤(워크플로우 선택 + 확인/취소) ───────────
def run_controls(p: dict, options: list[dict], key: str):
    """워크플로우 선택 selectbox + 확인/취소 버튼을 그리고, 선택 결과를 처리한다."""
    labels = [f"{w['name']} ({w['workflow_id']})" for w in options]
    default_idx = 0
    if p["matched"]:
        ids = [w["workflow_id"] for w in options]
        if p["workflow"]["workflow_id"] in ids:
            default_idx = ids.index(p["workflow"]["workflow_id"])
    pick = st.selectbox(
        "실행할 워크플로우 선택",
        range(len(options)),
        format_func=lambda i: labels[i],
        index=default_idx,
        key=f"sel_{key}",
    )
    matched_id = p["workflow"]["workflow_id"] if p["matched"] else None

    c1, c2 = st.columns(2)
    if c1.button("✅ 확인하고 실행", use_container_width=True, key=f"run_{key}"):
        chosen = options[pick]
        with st.spinner("API 호출 중..."):
            result = run_workflow_api(chosen["workflow_id"], p["query"])
        st.session_state.result = result
        # 사용자가 확인한 (요청 → 워크플로우) 매핑을 저장해 다음 매칭 정확도를 높임
        save_confirmed(chosen["workflow_id"], p["query"])
        # 매칭을 교정했다면(고른 게 다름) 원래 매칭은 오답 → negative 저장
        if matched_id and chosen["workflow_id"] != matched_id:
            save_rejected(matched_id, p["query"])
        # 대화 내역에 기록(단기 기억)
        st.session_state.history.append(
            {"question": p["query"], "answer": result.get("message", "실행됨"), "kind": "workflow"}
        )
        st.session_state.pending = None
        st.rerun()
    if c2.button("✖ 취소", use_container_width=True, key=f"cancel_{key}"):
        # 제안된 워크플로우가 틀려서 취소한 것으로 보고 negative 저장(오매칭 감소)
        if matched_id:
            save_rejected(matched_id, p["query"])
        st.session_state.pending = None
        st.rerun()


# ── 사이드바: 워크플로우 생성 / 목록 ──────────────────
with st.sidebar:
    st.subheader("➕ 워크플로우 생성")
    wf_name = st.text_input("워크플로우 명")
    wf_desc = st.text_area(
        "설명", height=100,
        placeholder="이 워크플로우가 어떤 요청을 처리하는지 적어주세요",
    )
    # 구분: system(시스템 제공) vs 일반(사용자 정의)
    CATEGORY_LABELS = {"general": "일반", "system": "시스템(system)"}
    wf_category = st.selectbox(
        "구분",
        options=list(CATEGORY_LABELS),
        format_func=lambda c: CATEGORY_LABELS[c],
    )
    if st.button("생성", use_container_width=True):
        if wf_name.strip() and wf_desc.strip():
            wf = create_workflow(wf_name, wf_desc, wf_category)
            st.success(f"생성 완료! id: {wf['workflow_id']}")
            st.rerun()
        else:
            st.warning("이름과 설명을 모두 입력해주세요.")

    st.divider()
    st.subheader(f"📋 등록된 워크플로우 ({count_workflows()})")
    for wf in list_workflows():
        badge = CATEGORY_LABELS.get(wf.get("category", "general"), "일반")
        st.write(f"**{wf['name']}**  `{badge}`")
        st.caption(f"`{wf['workflow_id']}` · {wf['description']}")

    st.divider()
    if st.button("🗑 대화 초기화", use_container_width=True):
        st.session_state.history = []
        st.session_state.pending = None
        st.session_state.result = None
        st.rerun()

# ── 대화 내역(단기 기억) 표시 ─────────────────────────
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("kind") == "workflow":
            st.caption("⚙️ 워크플로우 실행")

# ── 요청 입력 ──────────────────────────────────────────
query = st.chat_input("요청사항을 입력하세요...")
if query:
    with st.spinner("의도 분석 중..."):
        m = match_workflow(query)
    if m["matched"]:
        # 워크플로우 매칭됨 → 확인 절차로
        st.session_state.pending = {"query": query, **m}
        st.session_state.result = None
    else:
        # 어느 워크플로우에도 안 맞으면 LLM이 직접 답변(이전 대화 기억하여)
        with st.spinner("답변 생성 중..."):
            answer = general_answer(query, st.session_state.history)
        st.session_state.history.append({"question": query, "answer": answer, "kind": "llm"})
        st.session_state.pending = None
    st.rerun()

# ── 확인 절차 (워크플로우 매칭 시) ────────────────────
p = st.session_state.pending
if p:
    wf = p["workflow"]
    st.markdown(f"**요청:** {p['query']}")
    st.info(
        f"**'{wf['name']}'** 워크플로우로 실행하면 될까요?\n\n"
        f"`{wf['workflow_id']}` · 유사도 {p['score']:.2f} · {p['reason']}"
    )
    if p["score"] < config.ROUTE_THRESHOLD:
        st.warning("⚠️ 유사도가 낮습니다. 맞는지 한 번 더 확인해주세요.")
    if wf.get("penalized"):
        st.caption("ℹ️ 비슷한 요청이 과거에 거절된 적이 있어 순위가 낮춰진 상태입니다.")

    with st.expander("매칭 후보 보기"):
        for h in p["candidates"]:
            st.write(f"`{h['score']:.2f}` · **{h['name']}** · {h['workflow_id']}")

    run_controls(p, list_workflows(category="system"), key="matched")

# ── 실행 결과 ──────────────────────────────────────────
if st.session_state.result:
    st.success("워크플로우 API 호출 완료")
    st.json(st.session_state.result)
