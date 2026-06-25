"""매칭 실패 시 LLM이 직접 답변하는 fallback (단기 기억 포함)."""
import config
from clients import llm


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
