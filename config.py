"""환경변수 로드 및 앱 전역 설정값."""
import os

from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── 외부 서비스 설정 ───────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("COLLECTION")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))
EMBED_MODEL = os.getenv("EMBED_MODEL")
TOP_K = int(os.getenv("TOP_K"))

VLLM_URL = os.getenv("VLLM_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
VLLM_MODEL = os.getenv("VLLM_MODEL")

# ── 동작 파라미터 ──────────────────────────────────────
SCORE_THRESHOLD = 0.7   # 이 유사도를 넘는 검색 결과만 참고 자료로 사용
DEDUP_THRESHOLD = 0.92  # 이만큼 비슷하면 같은 질문으로 보고 기존 답변을 덮어씀(누적 개선)
TEMPERATURE = 0.7
MAX_TOKENS = 512
HISTORY_TURNS = 5       # LLM에게 기억시킬 최근 대화 턴 수(단기 기억)

VECTORS_CONFIG = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
