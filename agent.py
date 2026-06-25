"""LLM 에이전트: 도구(tool)를 LLM이 스스로 골라 호출하며 답을 만든다.

기존 generate_answer 가 '항상 검색 → 답변' 으로 흐름이 고정돼 있던 것과 달리,
여기서는 LLM이 매 턴 '지금 어떤 도구가 필요한지' 판단해서 호출한다.
"""
import json

import config
from clients import llm
from llm import generate_answer
from vectordb import count_points, search_similar

# ── 에이전트가 쓸 수 있는 도구 목록(스키마) ────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "벡터 DB에서 사용자 질문과 관련된 과거 Q&A·문서를 검색한다. "
                           "추천, 사실 확인, 이전에 저장된 정보가 필요할 때 사용하라.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 질문 또는 키워드"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_saved",
            "description": "현재 벡터 DB에 저장된 항목(Q&A·문서 조각) 수를 알려준다. "
                           "'몇 개 저장돼 있어?' 같은 질문에 사용하라.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """당신은 친절한 추천 도우미입니다.
필요하면 제공된 도구를 사용해 정보를 찾아 답하세요.
- 추천·사실·과거에 저장된 정보가 필요하면 search_knowledge 로 검색하세요.
- 저장된 항목 수를 물으면 count_saved 를 사용하세요.
- 도구가 필요 없는 일반 질문은 바로 답하세요.
도구 결과를 근거로 한국어로 친절하게 답변하세요."""


# ── 도구 실제 실행(디스패처) ───────────────────────────
def _run_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """도구를 실행해 (LLM에 돌려줄 문자열, UI 표시용 검색결과) 반환."""
    if name == "search_knowledge":
        results = search_similar(args.get("query", ""))
        if not results:
            return "검색 결과가 없습니다.", []
        text = "\n\n".join(f"(유사도 {r['score']:.2f}) {r['text']}" for r in results)
        return text, results
    if name == "count_saved":
        return f"현재 저장된 항목 수: {count_points()}개", []
    return f"알 수 없는 도구: {name}", []


# ── 에이전트 루프 ──────────────────────────────────────
def run_agent(query: str, history: list[dict] | None = None, max_steps: int = 5) -> dict:
    """LLM이 도구를 골라 호출하며 답을 만든다.

    반환: {"answer", "context"(검색결과), "steps"(에이전트가 한 일 로그)}
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-config.HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})

    collected_context: list[dict] = []
    steps: list[str] = []

    try:
        for _ in range(max_steps):
            resp = llm.chat.completions.create(
                model=config.VLLM_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            msg = resp.choices[0].message

            # 도구 호출이 없으면 = 최종 답변
            if not msg.tool_calls:
                return {"answer": msg.content, "context": collected_context, "steps": steps}

            # LLM이 도구를 요청 → 메시지에 그대로 기록
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 요청된 각 도구 실행 후 결과를 다시 메시지에 추가
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result, ctx = _run_tool(name, args)
                collected_context.extend(ctx)
                steps.append(f"🔧 {name}({args}) 호출")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # max_steps 초과 시: 도구 없이 마지막 정리 답변 요청
        resp = llm.chat.completions.create(
            model=config.VLLM_MODEL,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return {"answer": resp.choices[0].message.content, "context": collected_context, "steps": steps}

    except Exception:
        # 서버가 tool calling 미지원 등으로 실패하면 기존 고정 파이프라인으로 폴백
        context = search_similar(query)
        answer = generate_answer(query, context, history)
        return {"answer": answer, "context": context, "steps": ["⚠️ 에이전트 미지원 → 기본 검색 모드로 답변"]}
