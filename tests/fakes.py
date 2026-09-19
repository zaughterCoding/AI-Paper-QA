"""测试替身（test doubles）。

让测试不依赖外部世界：不下载模型、不调 LLM、不联网、不确定性地慢。

单独放一个模块，而不是散在各个测试文件里，是因为**替身一旦被复制成好几份，
它们就会各自漂移**，最后谁都不能代表真实组件的行为。Task 8 的检索测试、
Task 9 的问答测试都要用同一个替身。
"""

import math
import zlib

from app.core.config import EMBEDDING_DIM


class FakeEmbeddingClient:
    """确定性的假 embedding，形状与归一化都模仿真实客户端。

    实现是"词袋 + 哈希"：把每个词哈希到 384 维中的某一维上累加。这样
    **共享词汇的文本会得到更相似的向量**，检索测试因此能验证排序逻辑，
    而不是拿到一堆互不相关的随机数。

    它模仿不了的：**真正的语义**。它不知道 cat 和 kitten 相关，
    只知道它们没有共同的词。所以它只能验证"检索流程对不对"，
    不能验证"检索质量好不好"——后者要用真实模型跑离线评估（Task 14）。
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim

        for word in text.lower().split():
            # 必须用 crc32，不能用内置的 hash()：内置 hash 对字符串带进程级随机盐
            # （PYTHONHASHSEED），同一个词在不同进程里会落到不同维度上，
            # 测试就失去确定性、也没法复现了。
            bucket = zlib.crc32(word.encode("utf-8")) % self.dim
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector  # 空文本 / 全是未知字符 → 全零向量

        return [value / norm for value in vector]
