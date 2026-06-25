# 의미 기반 워크플로우 라우터 실습

사용자가 **워크플로우**(이름 + 설명)를 만들면 고유 id가 발급되고, 사용자 요청의 **의도(intent)** 를 분석해 가장 알맞은 워크플로우를 찾아 줍니다.
API를 호출하기 전에 **사용자 확인 절차**를 거치고, 확인된 경우에만 해당 워크플로우 id로 API를 호출하는 Streamlit 실습용 프로젝트입니다.

## 동작 흐름

```
워크플로우 생성 (이름·설명) → 고유 id 발급 → 임베딩해 벡터 DB 저장

요청 입력
  └─ 1. Qdrant에서 유사한 워크플로우 검색 (벡터 후보 좁히기)
  └─ 2. LLM이 후보 중 가장 적합한 워크플로우 id 확정
  └─ 3. "이 워크플로우 맞나요?" 사용자 확인 (직접 변경/취소 가능)
  └─ 4. 확인 시에만 해당 워크플로우 id로 API 호출
        · 확인/교정한 요청은 저장되어 매칭 정확도가 누적 향상
        · 취소/교정한 오매칭은 negative로 저장되어 다음 매칭에서 순위 하락
  └─ (어느 워크플로우에도 안 맞으면) LLM이 직접 답변
```

- **벡터 DB**: [Qdrant Cloud](https://qdrant.tech/) — 임베딩은 Qdrant Cloud Inference로 처리
- **임베딩 모델**: `sentence-transformers/all-MiniLM-L6-v2` (384차원, Cosine 거리)
- **LLM**: OpenAI 호환 vLLM 엔드포인트

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
WORKFLOW_COLLECTION=workflows
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_SIZE=384
```

> ⚠️ `.env`에는 실제 API 키가 들어가므로 깃에 커밋하지 마세요. (`.gitignore`에 이미 `.env` 등록됨)

## 실행

### 방법 1. 터미널에서 직접 실행

```bash
streamlit run router_app.py
```

브라우저에서 http://localhost:8501 접속.

### 방법 2. 노트북으로 실행

[run.ipynb](run.ipynb)를 열고 셀을 순서대로 실행하면 백그라운드에서 앱이 뜹니다. 종료는 마지막 셀(`proc.terminate()`)로 합니다.

## 사용법

1. 사이드바 **"➕ 워크플로우 생성"** 에서 이름·설명을 입력하고 **생성** → 고유 id 발급
2. 채팅창에 요청 입력 → 가장 알맞은 워크플로우가 매칭됨
3. **"이 워크플로우 맞나요?"** 확인 → 필요하면 직접 다른 워크플로우로 변경 가능
4. **확인하고 실행** 시에만 해당 워크플로우 id로 API 호출 (취소도 가능)

## 주요 설정값 (`config.py`)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ROUTE_THRESHOLD` | `0.5` | 매칭 유사도가 이 값 미만이면 '낮은 신뢰도' 경고 표시 |

## 동작 방식 (`workflows.py`)

- `create_workflow(name, description)` — 고유 id 발급 + 이름·설명을 임베딩해 벡터 DB 저장
- `match_workflow(query)` — 벡터 검색으로 후보를 좁히고 LLM이 가장 적합한 워크플로우 id 확정
- `run_workflow_api(workflow_id, request)` — 확인 후 해당 id로 가상 API 호출 (실행 시뮬레이션)

## 파일 구조

```
.
├── router_app.py  # Streamlit UI (진입점): 워크플로우 생성 + 요청 매칭 + 확인/실행
├── workflows.py   # 워크플로우 생성·조회 + 요청 매칭 + API 호출
├── config.py      # 환경변수 · 상수 설정
├── clients.py     # Qdrant / LLM 클라이언트 초기화
├── run.ipynb      # 앱을 백그라운드로 띄우고 종료하는 노트북
├── .env           # 환경 변수 (직접 작성)
└── README.md
```
