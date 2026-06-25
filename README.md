# 벡터 DB 추천 실습

질문하면 LLM이 답변하고, **사용자가 좋다고 평가한 답변은 벡터 DB에 쌓여서** 다음 유사 질문에 자동으로 활용되는 Streamlit 앱입니다. 간단한 RAG(검색 증강 생성) 흐름을 직접 만들어 보는 실습용 프로젝트입니다.

## 동작 흐름

```
질문 입력
  └─ 1. Qdrant에서 유사 질문 검색 (유사도 > 0.7 만 채택)
  └─ 2. 검색 결과를 참고 자료로 LLM이 답변 생성
  └─ 3. 사용자 피드백
        👍 → 질문·답변을 벡터 DB에 저장 (점점 똑똑해짐)
        👎 → 저장하지 않고 넘어감
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
pip install streamlit python-dotenv openai qdrant-client
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

## 주요 설정값 (`app.py`)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `SCORE_THRESHOLD` | `0.7` | 이 유사도를 넘는 검색 결과만 참고 자료로 사용 |
| `TEMPERATURE` | `0.7` | LLM 생성 다양성 |
| `MAX_TOKENS` | `512` | LLM 답변 최대 토큰 수 |
| `TOP_K` | `3` | 벡터 검색 시 가져올 결과 수 |

## 화면 구성

- **본문**: 채팅 형태의 질문/답변, DB 참고 시 유사도 표시
- **사이드바**: 저장된 Q&A 수 표시, `DB 초기화` 버튼으로 컬렉션 리셋

## 파일 구조

```
.
├── app.py        # Streamlit UI (진입점)
├── config.py     # 환경변수 · 상수 설정
├── clients.py    # Qdrant / LLM 클라이언트 초기화
├── vectordb.py   # 벡터 DB 검색 · 저장 · 문서 학습
├── llm.py        # LLM 답변 생성 · 피드백 개선
├── run.ipynb     # 앱을 백그라운드로 띄우고 종료하는 노트북
├── .env          # 환경 변수 (직접 작성)
└── README.md
```
