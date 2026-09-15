"""文本切分的测试。

切分是整条 RAG 链路的**第一步**，它一旦出错，后面全都错：
切错了，embedding 就是错的，检索也就是错的，而且这种错误很隐蔽——
系统照样能返回答案，只是答案质量变差，你不会收到任何报错。
所以这里测得比较细。
"""

import pytest

from app.rag.chunking import TextChunker


def test_chunker_returns_empty_list_for_empty_text():
    """空文本切出 0 个片段，而不是抛异常或返回空字符串的片段。"""
    assert TextChunker().chunk("") == []


def test_chunker_returns_empty_list_for_whitespace_only_text():
    """只有空白字符的文本，等同于空文本。"""
    assert TextChunker().chunk("   \n\t  ") == []


def test_chunker_splits_text_with_overlap():
    """切分主逻辑：chunk_size=4、overlap=1 时的预期切法。"""
    text = " ".join(str(i) for i in range(10))  # "0 1 2 3 4 5 6 7 8 9"
    chunks = TextChunker(chunk_size=4, overlap=1).chunk(text)

    assert [chunk.text for chunk in chunks] == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunker_returns_single_chunk_when_text_fits():
    """文本比 chunk_size 还短时，只切出一个片段。"""
    chunks = TextChunker(chunk_size=100, overlap=10).chunk("hello world")

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunk_indices_are_sequential_and_start_at_zero():
    """片段序号必须是 0,1,2,... —— Task 5 存库时靠它做唯一约束。"""
    text = " ".join(str(i) for i in range(20))
    chunks = TextChunker(chunk_size=4, overlap=1).chunk(text)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_token_count_matches_word_count_of_chunk_text():
    """token_count 必须和 text 里的词数自洽，不能对不上。"""
    text = " ".join(str(i) for i in range(20))
    chunks = TextChunker(chunk_size=5, overlap=2).chunk(text)

    for chunk in chunks:
        assert chunk.token_count == len(chunk.text.split())


def test_every_word_is_covered_by_some_chunk():
    """不变量：原文的每个词至少出现在一个片段里，不能丢内容。"""
    words = [f"w{i}" for i in range(37)]  # 用质数长度，避免整除掩盖边界问题
    chunks = TextChunker(chunk_size=8, overlap=3).chunk(" ".join(words))

    covered = {word for chunk in chunks for word in chunk.text.split()}
    assert covered == set(words)


def test_chunking_is_deterministic():
    """同样的输入必须切出同样的结果，否则测试和评测都无从谈起。"""
    text = " ".join(str(i) for i in range(50))
    chunker = TextChunker(chunk_size=10, overlap=3)

    assert [c.text for c in chunker.chunk(text)] == [c.text for c in chunker.chunk(text)]


def test_chunker_rejects_overlap_not_smaller_than_chunk_size():
    """overlap >= chunk_size 会导致切分不前进（死循环），必须直接拒绝。"""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=15)


def test_chunker_rejects_non_positive_chunk_size():
    """chunk_size <= 0 是非法配置，应该在构造时就报错，而不是切出垃圾。

    注意这两个用例都是 overlap **小于** chunk_size 的：
    (0, -1) 和 (-5, -10) 都能绕过「overlap >= chunk_size」那条检查，
    所以它们才能真正验证「chunk_size 必须为正」这层独立的保护有没有生效。
    """
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=-5, overlap=-10)
