"""환경변수 로드 및 앱 전역 설정값."""
import os

from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── 외부 서비스 설정 ───────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
WORKFLOW_COLLECTION = os.getenv("WORKFLOW_COLLECTION", "workflows")  # 사용자가 만든 워크플로우 저장용
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))
EMBED_MODEL = os.getenv("EMBED_MODEL")

VLLM_URL = os.getenv("VLLM_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
VLLM_MODEL = os.getenv("VLLM_MODEL")

# ── 동작 파라미터 ──────────────────────────────────────
ROUTE_THRESHOLD = 0.5   # 매칭 유사도가 이 값 미만이면 UI에서 '낮은 신뢰도' 경고 표시
DEDUP_THRESHOLD = 0.92  # 같은 워크플로우에 이만큼 유사한 학습 요청이 있으면 중복으로 보고 저장 생략
NEG_THRESHOLD = 0.80    # 과거 거절(negative)과 이만큼 유사하면 해당 워크플로우 순위를 낮춤
HISTORY_TURNS = 5       # LLM fallback 답변에 함께 넣을 최근 대화 턴 수(단기 기억)

VECTORS_CONFIG = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
