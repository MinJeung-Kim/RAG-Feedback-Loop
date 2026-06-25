"""Streamlit UI (진입점). 실제 로직은 vectordb / llm 모듈에 있음."""
import streamlit as st

from agent import run_agent
from llm import refine_answer
from vectordb import (
    count_points,
    find_existing_qa,
    read_uploaded_file,
    reset_collection,
    save_document,
    save_to_db,
)

# ── 헤더 ───────────────────────────────────────────────
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
    st.metric("저장된 Q&A 수", count_points())

    if st.button("DB 초기화"):
        reset_collection()
        st.session_state.history = []
        st.session_state.pending = None
        st.success("초기화 완료!")
        st.rerun()

    # ── 문서 학습 ──────────────────────────────────────
    st.divider()
    st.subheader("📄 문서 학습")
    st.caption("문서를 미리 넣어두면 LLM이 답변할 때 참고합니다.")

    uploaded = st.file_uploader("파일 업로드 (txt/md/pdf)", type=["txt", "md", "pdf"])
    pasted = st.text_area("또는 텍스트 붙여넣기", height=120)

    if st.button("학습시키기", use_container_width=True):
        doc_text, source = "", ""
        if uploaded is not None:
            doc_text = read_uploaded_file(uploaded)
            source = uploaded.name
        elif pasted.strip():
            doc_text = pasted
            source = "붙여넣은 텍스트"

        if doc_text.strip():
            with st.spinner("문서를 임베딩해서 저장 중..."):
                n = save_document(doc_text, source)
            st.success(f"'{source}' → {n}개 조각으로 저장 완료!")
            st.rerun()
        else:
            st.warning("업로드하거나 붙여넣은 문서가 없어요.")

# ── 대화 히스토리 출력 ────────────────────────────────
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        if item.get("from_db"):
            st.caption(f"DB 참고 (유사도 {item['top_score']:.2f})")
        if item.get("steps"):
            with st.expander("🤖 에이전트가 한 일"):
                for s in item["steps"]:
                    st.write(s)

# ── 피드백 UI (답변 직후) ─────────────────────────────
if st.session_state.pending:
    p = st.session_state.pending
    with st.chat_message("user"):
        st.write(p["question"])
    with st.chat_message("assistant"):
        st.write(p["answer"])
        if p.get("from_db"):
            st.caption(f"DB 참고 (유사도 {p['top_score']:.2f})")
        if p.get("steps"):
            with st.expander("🤖 에이전트가 한 일"):
                for s in p["steps"]:
                    st.write(s)

    st.markdown("**이 답변, 구체적으로 알려주면 더 똑똑해져요.**")
    good = st.text_area("👍 도움이 된 점", placeholder="예: 단계별 설명이 이해하기 쉬웠어요", height=80)
    bad = st.text_area("👎 아쉽거나 틀린 점", placeholder="예: 가격 정보가 빠졌고, 2번 항목이 사실과 달라요", height=80)

    col1, col2, col3 = st.columns(3)

    with col1:
        # 피드백을 반영해 답변을 개선한 뒤 '개선된 답변'을 저장
        if st.button("✏️ 피드백 반영·개선 저장", use_container_width=True):
            if not (good.strip() or bad.strip()):
                st.warning("도움된 점 또는 아쉬운 점을 한 가지라도 적어주세요.")
            else:
                with st.spinner("피드백을 반영해 답변을 다듬는 중..."):
                    improved = refine_answer(p["question"], p["answer"], good, bad)
                feedback_note = f"[좋음] {good.strip()}\n[아쉬움] {bad.strip()}".strip()
                # 같은 질문이 이미 있으면 덮어써서 누적 개선, 없으면 새로 저장
                existing_id = find_existing_qa(p["question"])
                save_to_db(p["question"], improved, feedback=feedback_note, point_id=existing_id)
                p["answer"] = improved  # 화면·히스토리에도 개선된 답변 반영
                p["saved"] = True
                st.session_state.history.append(p)
                st.session_state.pending = None
                if existing_id:
                    st.success("기존 답변을 피드백으로 더 개선해 갱신했어요! (누적 학습)")
                else:
                    st.success("피드백을 반영해 개선된 답변을 저장했어요! 다음 유사 질문에 활용됩니다.")
                st.rerun()

    with col2:
        # 피드백 없이 지금 답변 그대로 저장
        if st.button("👍 그대로 저장", use_container_width=True):
            save_to_db(p["question"], p["answer"])
            p["saved"] = True
            st.session_state.history.append(p)
            st.session_state.pending = None
            st.success("DB에 저장됐어요!")
            st.rerun()

    with col3:
        if st.button("건너뛰기", use_container_width=True):
            p["saved"] = False
            st.session_state.history.append(p)
            st.session_state.pending = None
            st.rerun()

# ── 입력창 ────────────────────────────────────────────
if not st.session_state.pending:
    query = st.chat_input("무엇이든 물어보세요...")
    if query:
        with st.spinner("에이전트가 생각하는 중..."):
            result = run_agent(query, st.session_state.history)

        context = result["context"]
        st.session_state.pending = {
            "question": query,
            "answer": result["answer"],
            "from_db": len(context) > 0,
            "top_score": max((c["score"] for c in context), default=0.0),
            "steps": result["steps"],
        }
        st.rerun()
