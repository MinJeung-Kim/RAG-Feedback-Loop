"""LLM 답변 생성 및 피드백 기반 답변 개선."""
import config
from clients import llm


def generate_answer(query: str, context: list[dict], history: list[dict] | None = None) -> str:
    if context:
        ctx_text = "\n\n".join([c["text"] for c in context])
        # 과거에 사용자가 지적한 '아쉬운 점'을 모아 같은 실수를 반복하지 않게 가드로 주입
        notes = [c["feedback"] for c in context if c.get("feedback")]
        guard = ""
        if notes:
            guard = "\n\n[과거 피드백 — 아래 지적을 반영하고 같은 실수를 반복하지 마세요]\n" + "\n".join(notes)
        system = f"""당신은 친절한 추천 도우미입니다.
아래 참고 자료를 바탕으로 사용자 질문에 답하세요.
참고 자료가 없으면 일반 지식으로 답하세요.

[참고 자료]
{ctx_text}{guard}"""
    else:
        system = "당신은 친절한 추천 도우미입니다. 사용자 질문에 성실하게 답하세요."

    # 최근 대화 턴을 메시지에 넣어 단기 기억(맥락) 제공
    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-config.HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})

    resp = llm.chat.completions.create(
        model=config.VLLM_MODEL,
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return resp.choices[0].message.content


def refine_answer(question: str, answer: str, good: str, bad: str) -> str:
    """사용자의 구체적인 피드백을 반영해 답변을 다시 다듬는다."""
    system = """당신은 답변을 개선하는 편집자입니다.
사용자가 알려준 '도움이 된 점'은 살리고, '아쉬운/틀린 점'은 고쳐서
질문에 대한 더 나은 답변을 새로 작성하세요.
설명이나 머리말 없이 개선된 답변 본문만 출력하세요."""
    user_msg = f"""[질문]
{question}

[기존 답변]
{answer}

[도움이 된 점]
{good or "(없음)"}

[아쉬운/틀린 점]
{bad or "(없음)"}"""

    resp = llm.chat.completions.create(
        model=config.VLLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return resp.choices[0].message.content
