"""把文本转成向量。

这是 RAG 的"翻译层"：它把人和模型都能理解的文本，翻译成只有数学能比较的向量。
检索之所以可能，全靠这一步——两段文本的语义距离，变成了两个向量之间的余弦距离。
"""

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingClient:
    """对 sentence-transformers 的一层薄封装。

    为什么要包一层而不是到处直接调 SentenceTransformer？三个原因：

    1. **换实现时只改一处**：将来换成 OpenAI 的 embedding API、或者换成
       bge/ e5 系列模型，改动都局限在这个文件里，上层不用动。
    2. **测试可以替换**：tests/fakes.py 里的假客户端实现同样的两个方法，
       测试因此不需要下载模型（见 tests/test_embeddings.py 的契约测试）。
    3. **把"用不用归一化"这类决定固定下来**：它不该由每个调用方各自决定。

    注意模型是在 `__init__` 里加载的——这一步要几秒、占几百 MB 内存。
    所以**调用方必须复用同一个实例**，不能每次请求都 new 一个。
    复用机制就是下面的 `get_embedding_client()`。
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """把一段文本变成向量。"""
        # 故意委托给 embed_texts，而不是单独调一次 encode：
        # 两条独立的代码路径迟早会漂移（一个忘了归一化、一个忘了 tolist），
        # 单条路径从结构上杜绝这种可能。
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量转换。批量比循环单条快得多——模型能一次处理一整批。"""
        if not texts:
            # 提前返回，不把空列表送进模型：某些版本对空输入会直接抛异常，
            # 而"没有文本"是完全正常的业务情况（比如某个文档没有任何片段）。
            return []

        vectors = self.model.encode(texts, normalize_embeddings=True)
        # encode 返回 numpy 数组。转成 list[float] 有两个理由：
        # 一是让返回值脱离 numpy（上层不需要为了拿个向量而依赖 numpy），
        # 二是 numpy 的 float32 直接塞进 JSON 会报错，转成 Python float 才能序列化。
        return [vector.tolist() for vector in vectors]


# 进程级唯一实例。前面的下划线表示"这是模块内部状态，别直接碰它"。
_default_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """拿到全局唯一的 EmbeddingClient（没有就创建一个）。

    为什么必须是单例：`EmbeddingClient.__init__` 要加载模型——几秒 + 几百 MB。
    如果每个请求都 new 一个，第一次并发就会把内存吃光，而且每次请求都要先等几秒。

    为什么用模块级变量而不是 `functools.lru_cache`：
    效果一样，但模块级变量的意图更直白（"这里有一个全局的重对象"），
    而且测试里可以直接重置它。

    为什么不做成 FastAPI 的依赖函数、放在 api/ 目录下：
    它其实是个**普通的零参数函数**——FastAPI 的 `Depends` 接受任何这样的可调用对象，
    不需要 import fastapi。放在这里，`rag/` 层保持"纯逻辑、不认识 Web 框架"，
    而重对象的生命周期归它自己的模块管。

    这是**惰性**的：import 时不加载模型，第一次真正需要时才加载。
    所以 pytest 收集用例、或只跑不涉及 embedding 的测试时，没有任何模型开销。
    """
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient(get_settings().embedding_model)
    return _default_client
