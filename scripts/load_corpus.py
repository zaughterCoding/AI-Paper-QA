"""把 eval/corpus/ 里的语料通过 HTTP 接口导入数据库。

用法（先在另一个终端把服务跑起来）：

    E:\\conda-envs\\paperqa\\python.exe -m uvicorn app.main:app --port 8000
    E:\\conda-envs\\paperqa\\python.exe scripts\\load_corpus.py

## 为什么走 HTTP，而不是直接调 IngestService

直接 `SessionLocal()` + `IngestService(...)` 会短得多，也不用先起服务。
但那样就绕开了这条链路上最容易出问题的一环：

    路由 → 依赖注入 → 真实 EmbeddingClient → IndexingService → 数据库

Task 8b 之前，`POST /documents` 的"导入即索引"这段**从来没有用真模型跑过**——
所有 API 测试都用 `FakeEmbeddingClient` 替换掉了真客户端（必须替换，
否则每个测试都要加载几百 MB 的模型）。也就是说，单测全绿，但这条路径
在真实输入上是不是通的，没人知道。

这个脚本就是来跑那一次的。它顺便验证了几件单测覆盖不到的事：
真模型加载、真编码耗时、事务在真数据量下的行为、以及响应里的
`embedded_chunk_count` 在真语料上是不是合理。

## 幂等性

重复跑不会产生重复数据：`POST /documents` 按 content_hash 去重，
第二次导入同一篇会返回 `created=false`、`embedded_chunk_count=0`。
（这个去重行为本身也是导入功能的一部分，不是这个脚本做的。）
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parents[1] / "eval" / "corpus"


def load_manifest() -> list[dict]:
    path = CORPUS_DIR / "sources.json"
    if not path.exists():
        raise SystemExit(f"找不到 {path}，先跑 scripts/fetch_corpus.py 抓语料")
    return json.loads(path.read_text(encoding="utf-8"))["papers"]


def main() -> int:
    parser = argparse.ArgumentParser(description="把语料导入数据库")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    papers = load_manifest()

    # 超时给得很宽：第一个请求要加载 embedding 模型（几百 MB，几秒到几十秒），
    # 而且每篇论文要编码几百个片段。默认的 5 秒会在第一篇上直接超时。
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0)

    failures = 0
    with httpx.Client(base_url=args.base_url, timeout=timeout) as client:
        for paper in papers:
            path = CORPUS_DIR / paper["file"]
            payload = {
                "title": paper["title"],
                "source": paper["url"],
                "content": path.read_text(encoding="utf-8"),
            }

            try:
                response = client.post("/documents", json=payload)
            except httpx.HTTPError as error:
                print(f"  失败  {paper['slug']:32} {type(error).__name__}: {error}")
                failures += 1
                continue

            if response.status_code != 201:
                # 打印响应体：接口的 422 里带着具体的校验错误，
                # 只报状态码的话还得手工复现一次才知道哪里不对。
                print(f"  失败  {paper['slug']:32} HTTP {response.status_code}  {response.text[:200]}")
                failures += 1
                continue

            body = response.json()
            state = "新建" if body["created"] else "已存在"
            print(
                f"  {state}  {paper['slug']:32} "
                f"{body['chunk_count']:>4} 片段，本次编码 {body['embedded_chunk_count']:>4}"
            )

    print()
    if failures:
        print(f"⚠️  {failures} 篇导入失败")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
