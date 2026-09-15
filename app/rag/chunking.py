"""把长文档切成带重叠的小片段。

为什么必须切分？两个硬限制：

1. **embedding 模型有输入长度上限。** 一篇论文塞不进去，塞进去也会被截断，
   截断意味着后面的内容等于没有。
2. **检索需要"精确命中"。** 如果一个片段里混着五六个主题，检索到它也没用——
   你不知道是冲着哪部分来的。片段切得越聚焦，检索越准。

为什么片段之间要**重叠**？因为句子会被切断：

    片段 A: ... self-attention solves the problem of long-range
    片段 B: dependencies by relating all positions ...

如果 A 和 B 不重叠，那么"long-range dependencies"这个关键概念被劈成两半，
两边都检索不到它。留一段重叠（默认 30 个词）就是为了让这种跨边界的表达
至少在某一个片段里是完整的。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """一个文本片段。frozen=True 让它不可变——片段一旦切出来就不该被改动。

    用 dataclass 而不是普通类：它自动生成 __init__、__eq__、__repr__，
    而且 frozen 之后可以安全地放进 set、当 dict 的 key。
    """

    index: int  # 在所属文档内的序号，从 0 开始
    text: str
    token_count: int  # 近似 token 数，见 TextChunker 的说明


class TextChunker:
    """按词切分文本，相邻片段保留 overlap 个词的重叠。

    关于 token_count：这里用**词数**近似 token 数。真正的 token 化要跑
    模型自带的分词器，成本高。对 all-MiniLM-L6-v2 这类模型，
    英文里 1 个词 ≈ 1.3 个 token，所以词数是个够用的近似值——
    它的用途只是"估算片段大小"，不参与任何精确计算。
    """

    def __init__(self, chunk_size: int = 180, overlap: int = 30) -> None:
        # 这两个校验必须在构造时就做，而不是等到 chunk() 里。
        # 理由：配置错误应该**尽早、尽响亮地**失败，而不是等到处理了几百篇
        # 文档之后才在某个片段上暴露出奇怪的结果。
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must not be negative")
        if overlap >= chunk_size:
            # 如果 overlap 不小于 chunk_size，切片起点就不会前进：
            # 下一次的 start = end - overlap <= start，循环原地打转。
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """切分文本。空文本（或只有空白字符）返回空列表。"""
        # split() 不带参数时会按任意空白切分并丢弃空串，
        # 所以 "   \n\t " 会得到 []，和空字符串的处理天然统一。
        words = text.split()
        if not words:
            return []

        chunks: list[TextChunk] = []
        start = 0
        index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]

            chunks.append(
                TextChunk(
                    index=index,
                    text=" ".join(chunk_words),
                    token_count=len(chunk_words),
                )
            )

            # 已经切到结尾了，结束。没有这一步，最后一段会因为
            # 再次计算 start 而重复切出一段内容完全被上一段包含的片段。
            if end == len(words):
                break

            # 回退 overlap 个词，制造重叠。因为 overlap < chunk_size，
            # start 每次严格增大，循环必然终止。
            start = end - self.overlap
            index += 1

        return chunks
