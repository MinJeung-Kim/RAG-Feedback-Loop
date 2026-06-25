# 벡터 DB 추천 실습

질문하면 **LLM 에이전트**가 필요할 때 벡터 DB를 검색해 답하고, 사용자의 **구체적인 피드백**으로 답변을 다듬어 DB에 쌓아 **점점 똑똑해지는** Streamlit 앱입니다. RAG(검색 증강 생성) → 피드백 학습 → 에이전트로 발전시켜 보는 실습용 프로젝트입니다.

## 주요 기능

| 기능 | 설명 | 관련 파일 |
| --- | --- | --- |
| 🤖 **에이전트** | LLM이 상황에 맞게 도구(`search_knowledge`, `count_saved`)를 **스스로 골라** 호출 | `agent.py` |
| 🔍 **RAG 검색** | 질문과 유사한 과거 Q&A·문서를 벡터 DB에서 검색해 답변 근거로 사용 | `vectordb.py` |
| ✍️ **피드백 학습** | "도움된 점/아쉬운 점" 피드백을 반영해 답변을 다시 다듬어 저장 | `llm.py` |
| ♻️ **누적 개선** | 같은 질문이 또 오면 새로 쌓지 않고 **기존 답변을 덮어써** 품질 누적 | `vectordb.py` |
| 🚫 **실수 방지** | 과거 '아쉬운 점' 피드백을 답변 생성 시 가드로 주입해 같은 실수 반복 방지 | `llm.py` |
| 📄 **문서 학습** | txt/md/pdf 파일이나 텍스트를 chunk로 쪼개 임베딩·저장 | `vectordb.py` |
| 💬 **단기 기억** | 최근 대화 5턴을 LLM에 전달해 후속 질문의 맥락 유지 | `llm.py`, `agent.py` |

## 동작 흐름

```
질문 입력
  └─ 1. 에이전트(LLM)가 판단: 도구가 필요한가?
        ├─ 추천·사실·과거 정보 필요  → search_knowledge (벡터 DB 검색)
        ├─ "몇 개 저장돼 있어?"       → count_saved (저장 개수 조회)
        └─ 일반 질문(예: 1+1)        → 도구 없이 바로 답변
  └─ 2. (검색 시) 유사도 > 0.7 결과 + 과거 피드백 가드를 근거로 답변 생성
  └─ 3. 사용자 피드백
        ✏️ 피드백 반영·개선 저장 → 답변을 다듬어 저장 (같은 질문이면 덮어쓰기)
        👍 그대로 저장          → 현재 답변 그대로 저장
        🚶 건너뛰기             → 저장하지 않음
```

> 벡터 DB에는 **Q&A**(`question`/`answer`/`feedback`)와 **문서 조각**(`text`/`source`) 두 종류가 같은 컬렉션에 저장되며, 검색은 둘 다 한 번에 다룹니다.

### 기억의 두 종류

| 종류 | 범위 | 구현 |
| --- | --- | --- |
| **장기 기억** | 세션이 끝나도 유지 | 벡터 DB (Q&A · 문서 · 피드백) |
| **단기 기억** | 같은 대화 안에서만 | 최근 5턴 대화 히스토리 (`HISTORY_TURNS`) |

- **벡터 DB**: [Qdrant Cloud](https://qdrant.tech/) — 임베딩은 Qdrant Cloud Inference로 처리
- **임베딩 모델**: `sentence-transformers/all-MiniLM-L6-v2` (384차원, Cosine 거리)
- **LLM**: OpenAI 호환 vLLM 엔드포인트 (에이전트용 **tool calling** 지원 필요)

## 요구 사항

- Python 3.10+
- Qdrant Cloud 계정 및 API 키
- OpenAI 호환 LLM 엔드포인트(vLLM 등)

## 설치

```bash
# 가상환경 생성 및 활성화 (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
# 또는 개별 설치
pip install streamlit python-dotenv openai qdrant-client pypdf
```

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 값을 채웁니다.

```dotenv
# Qdrant 설정
QDRANT_URL=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>

# LLM 설정 (OpenAI 호환 / vLLM)
VLLM_URL=http://<your-llm-host>/v1
VLLM_API_KEY=<your-llm-api-key>
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B

# 컬렉션 및 임베딩 설정
COLLECTION=qa_collection
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_SIZE=384
TOP_K=3
```

> ⚠️ `.env`에는 실제 API 키가 들어가므로 깃에 커밋하지 마세요. (`.gitignore`에 `.env` 추가 권장)

## 실행

### 방법 1. 터미널에서 직접 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속.

### 방법 2. 노트북으로 실행

[run.ipynb](run.ipynb)를 열고 셀을 순서대로 실행하면 백그라운드에서 앱이 뜹니다. 종료는 마지막 셀(`proc.terminate()`)로 합니다.

## 에이전트(tool calling) 서버 설정

`agent.py`는 OpenAI 호환 **tool calling** 기능을 사용합니다. vLLM 서버를 이 기능과 함께 띄워야 에이전트가 도구를 호출할 수 있습니다.

```bash
vllm serve Qwen/Qwen3.6-35B-A3B --enable-auto-tool-choice --tool-call-parser hermes
```

> 서버가 tool calling을 지원하지 않으면, 에이전트는 자동으로 **기존 "검색 → 답변" 방식으로 폴백**합니다 (화면에 "⚠️ 에이전트 미지원" 로그 표시).

## 주요 설정값 (`config.py`)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `SCORE_THRESHOLD` | `0.7` | 이 유사도를 넘는 검색 결과만 참고 자료로 사용 |
| `DEDUP_THRESHOLD` | `0.92` | 이만큼 비슷하면 같은 질문으로 보고 기존 답변을 덮어씀(누적 개선) |
| `TEMPERATURE` | `0.7` | LLM 생성 다양성 |
| `MAX_TOKENS` | `512` | LLM 답변 최대 토큰 수 |
| `HISTORY_TURNS` | `5` | LLM에 전달할 최근 대화 턴 수(단기 기억) |
| `TOP_K` | `3` | 벡터 검색 시 가져올 결과 수 (`.env`) |

## 화면 구성

- **본문**: 채팅 형태의 질문/답변, DB 참고 시 유사도 표시, "🤖 에이전트가 한 일" 패널로 호출한 도구 확인
- **답변 직후**: 피드백 입력칸(도움된 점/아쉬운 점) + 저장 버튼 3종
- **사이드바**: 저장 항목 수 표시, `DB 초기화` 버튼, `📄 문서 학습`(txt/md/pdf 업로드·텍스트 붙여넣기)

## 파일 구조

```
.
├── app.py        # Streamlit UI (진입점)
├── config.py     # 환경변수 · 상수 설정
├── clients.py    # Qdrant / LLM 클라이언트 초기화
├── vectordb.py   # 벡터 DB 검색 · 저장 · 문서 학습
├── llm.py        # LLM 답변 생성 · 피드백 개선
├── agent.py      # LLM 에이전트 (도구를 스스로 골라 호출)
├── run.ipynb     # 앱을 백그라운드로 띄우고 종료하는 노트북
├── requirements.txt
├── .env          # 환경 변수 (직접 작성)
└── README.md
```

### 모듈 의존 관계 (순환 없음)

```
config  ←  clients  ←  vectordb  ←┐
                    ←  llm        ←┤
                                   ├─  agent  ←  app
                    (vectordb·llm) ←┘
```

- `config` : 환경변수·상수 (의존 없음)
- `clients` : `qdrant`·`llm` 공유 인스턴스 생성
- `vectordb` / `llm` : 검색·저장 / 답변 생성 같은 핵심 로직
- `agent` : 위 로직을 **도구**로 묶어 LLM이 골라 쓰게 함
- `app` : Streamlit UI (진입점) — 로직은 모두 위 모듈에 위임
