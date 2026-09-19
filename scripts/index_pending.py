"""给全库还没有向量的片段补上向量（离线批处理入口）。

和 db.py 一样，用 conda 环境的 python 直接运行：

    E:\\conda-envs\\paperqa\\python.exe scripts\\index_pending.py

（项目是 editable 安装的，`app` 包在任何目录都能 import，
所以这里不需要像 `-m` 那样操心 sys.path。）

什么时候需要它：

1. **历史数据**：在索引功能接进 `POST /documents` 之前导入的文档，
   片段全是 NULL 向量，检索永远看不到它们。
2. **补漏**：某次导入在索引阶段失败了（文档存下了、向量没生成）。
3. **换模型之后**：换了 embedding 模型，全部语料都要重新编码。
   ⚠️ 注意这个脚本目前只处理 **embedding 为 NULL** 的片段，
   已经有向量的**不会被覆盖**——换模型的重建需要另写一个覆盖式的脚本，
   否则新旧模型的向量会混在同一个空间里，检索结果变成噪音。
   （这是有意留的缺口，见 docs/Findings_and_Decisions.md。）

这是**一次性脚本**，不是常驻服务：它自己开 session、自己收尾，
跑完就退出。
"""

import sys

from app.core.database import SessionLocal
from app.rag.embeddings import get_embedding_client
from app.services.indexing import IndexingService


def main() -> int:
    session = SessionLocal()
    try:
        # 注意这一行会**加载模型**（几秒、几百 MB）。这是脚本，等得起。
        service = IndexingService(session, get_embedding_client())
        result = service.index_all_pending()
    finally:
        session.close()

    print(f"已编码 {result.embedded_count} 个片段，跳过 {result.skipped_count} 个（已有向量）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
