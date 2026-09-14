from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """存活探针：只回答「服务进程还在不在」，不检查数据库。"""
    return {"status": "ok"}
