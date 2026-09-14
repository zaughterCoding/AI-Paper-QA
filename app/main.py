from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。

    用工厂函数而不是模块级 app，是为了让测试可以反复创建互相隔离的应用实例。
    """
    app = FastAPI(title="AI Paper QA Backend")
    app.include_router(router)
    return app


app = create_app()
