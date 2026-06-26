"""확정된 워크플로우 id로 (가상) API를 호출하는 실행 계층."""
from workflows import list_workflows


def run_workflow_api(workflow_id: str, request: str) -> dict:
    """확정된 워크플로우 id로 가상 API를 호출한다 (실행 시뮬레이션)."""
    wf = next((w for w in list_workflows() if w["workflow_id"] == workflow_id), None)
    if not wf:
        return {"status": "error", "message": f"워크플로우 {workflow_id} 를 찾을 수 없습니다."}
    return {
        "status": "executed",
        "endpoint": f"POST /api/workflows/{workflow_id}/execute",
        "workflow_id": workflow_id,
        "workflow_name": wf["name"],
        "request": request,
        "message": f"워크플로우 '{wf['name']}'를 실행했습니다.",
    }
