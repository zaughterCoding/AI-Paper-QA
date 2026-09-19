"""Embedding 客户端的测试。

分两类，目的不同：

1. **契约测试**：同一组断言分别跑在假客户端和真客户端上。这样"假货"不会
   悄悄跑偏——如果哪天假的维度错了、归一化忘了，契约测试会同时对两边报警。
2. **真实模型测试**：默认跳过（加载模型慢，且需要联网下载权重），
   设 `PAPERQA_RUN_MODEL_TESTS=1` 才跑。
"""

import math
import os
from functools import lru_cache

import pytest

from app.core.config import EMBEDDING_DIM, get_settings
from app.rag.embeddings import EmbeddingClient
from tests.fakes import FakeEmbeddingClient

RUNS_REAL_MODEL = os.environ.get("PAPERQA_RUN_MODEL_TESTS") == "1"
SKIP_REASON = "真实模型测试默认关闭；设 PAPERQA_RUN_MODEL_TESTS=1 开启"


@lru_cache(maxsize=1)
def _real_client() -> EmbeddingClient:
    """整个测试会话只加载一次模型。加载一次要几秒、占几百 MB，
    每个测试都新建一个实例是不可接受的。"""
    return EmbeddingClient(get_settings().embedding_model)


def _make_fake() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


def _make_real() -> EmbeddingClient:
    if not RUNS_REAL_MODEL:
        pytest.skip(SKIP_REASON)
    return _real_client()


@pytest.fixture(params=[pytest.param(_make_fake, id="fake"), pytest.param(_make_real, id="real")])
def client(request: pytest.FixtureRequest):
    """同一组测试跑两遍：一遍假客户端，一遍真模型。"""
    return request.param()


@pytest.fixture(scope="session")
def real_client() -> EmbeddingClient:
    if not RUNS_REAL_MODEL:
        pytest.skip(SKIP_REASON)
    return _real_client()


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。两个向量都已归一化，所以点积就是余弦。"""
    return sum(x * y for x, y in zip(a, b))


# --- 契约测试：假客户端和真客户端都必须满足 ---------------------------------


def test_embed_text_returns_vector_of_schema_dimension(client) -> None:
    """维度必须是 384——因为 chunks.embedding 列的类型就是 vector(384)。

    这个断言把组件和数据库 schema 绑在一起：如果哪天换了模型导致维度变化，
    这里会先报错，而不是等到写库时才炸出一个看不懂的 pgvector 类型错误。
    """
    vector = client.embed_text("self attention mechanism")

    assert isinstance(vector, list)
    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(value, float) for value in vector)


def test_vectors_are_normalized(client) -> None:
    """归一化后向量长度是 1。

    为什么要归一化：pgvector 的余弦距离 `<=>` 和点积 `<#>` 在单位向量上等价，
    而且归一化能避免"长文本的向量天然更长"这种长度偏差影响相似度排序。
    """
    vector = client.embed_text("self attention mechanism")

    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_texts_returns_one_vector_per_text_in_order(client) -> None:
    vectors = client.embed_texts(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    # 用 approx 而不是 ==：真实模型批量编码时的 padding 可能让
    # 最后几位浮点数与单独编码有极小差异
    assert vectors[0] == pytest.approx(client.embed_text("alpha"), abs=1e-6)
    assert vectors[2] == pytest.approx(client.embed_text("gamma"), abs=1e-6)


def test_embed_texts_of_empty_list_returns_empty(client) -> None:
    """空列表不能走到模型里。

    这个提前返回不只是省时间：某些版本的模型对空输入会直接抛异常，
    而"没有文本"是完全正常的业务情况（比如某个文档没有任何片段）。
    """
    assert client.embed_texts([]) == []


def test_same_text_gives_same_vector(client) -> None:
    assert client.embed_text("deterministic") == client.embed_text("deterministic")


# --- 假客户端自己的契约 ------------------------------------------------------


def test_fake_client_similarity_follows_shared_words() -> None:
    """把假客户端的能力边界写成测试，免得有人误以为它有语义。

    它只认词汇重叠，不认意思。所以：
    - 能用来测"排序逻辑对不对"（共享词的应该排前面）
    - 不能用来测"检索准不准"（它认为 cat 和 kitten 毫无关系）
    """
    fake = FakeEmbeddingClient()

    base = fake.embed_text("self attention mechanism")
    related = fake.embed_text("self attention layer")
    unrelated = fake.embed_text("banana bread recipe")

    assert _cosine(base, related) > _cosine(base, unrelated)


# --- 真实模型专有：验证它确实理解语义 ---------------------------------------


def test_real_model_places_related_words_closer(real_client: EmbeddingClient) -> None:
    """真模型的核心价值：意思相近的文本，向量也相近。

    这是假客户端永远做不到的事，也是整个 RAG 检索能工作的前提。
    """
    cat = real_client.embed_text("cat")
    kitten = real_client.embed_text("kitten")
    airplane = real_client.embed_text("airplane")

    assert _cosine(cat, kitten) > _cosine(cat, airplane)


def test_real_model_matches_paragraph_when_words_overlap(
    real_client: EmbeddingClient,
) -> None:
    """问题和对应段落共用词汇时，相似度排序是正确的。

    这是检索能工作的最低前提：至少"问什么词、就命中讲那个词的段落"。
    """
    question = real_client.embed_text("What is self-attention?")
    relevant = real_client.embed_text(
        "Self-attention relates all positions in a sequence to compute its representation."
    )
    irrelevant = real_client.embed_text("We trained the model for three days on eight GPUs.")

    assert _cosine(question, relevant) > _cosine(question, irrelevant)


def test_real_model_matches_chunk_when_question_is_paraphrased(
    real_client: EmbeddingClient,
) -> None:
    """最重要的一个测试：问题换说法、与相关 chunk 不共用关键词时，仍能命中。

    这才是 Task 8/9 真实发生的事情——检索的**不是句子，是 chunk**（Task 4 切出来的
    多句片段）。用单句测检索会严重低估系统能力，因为单句词汇面太窄，
    一个共享词就足以翻盘（见下面那个特征化测试）。

    实测（2026-09-18，multi-qa-MiniLM-L6-cos-v1）：

        问句   "How does the model handle long-range dependencies?"
        相关   "The Transformer follows an encoder-decoder structure ...
                Self-attention relates the different positions of a single
                sequence ... capture dependencies between distant positions ..."  → 0.393
        无关   "We trained the base models for 100,000 steps or 12 hours on
                eight NVIDIA P100 GPUs. ..."                                       → 0.252

    问句里的 long-range / dependencies 在相关 chunk 里**一个都没出现**，
    但模型仍然把它排在前面——这才是 embedding 相对关键词检索的价值所在。
    """
    question = real_client.embed_text("How does the model handle long-range dependencies?")
    attention_chunk = real_client.embed_text(
        "The Transformer follows an encoder-decoder structure using stacked self-attention "
        "and point-wise, fully connected layers. Self-attention relates the different positions "
        "of a single sequence in order to compute a representation of that sequence. This allows "
        "the model to capture dependencies between distant positions regardless of their distance "
        "in the sequence, unlike recurrent networks which must process the input sequentially."
    )
    training_chunk = real_client.embed_text(
        "We trained the base models for a total of 100,000 steps or 12 hours on eight NVIDIA P100 "
        "GPUs. We used the Adam optimizer with a custom learning rate schedule that increases "
        "linearly for the first 4000 steps and then decays proportionally to the inverse square "
        "root of the step number. Dropout was applied to the output of each sub-layer."
    )

    assert _cosine(question, attention_chunk) > _cosine(question, training_chunk)


def test_real_model_gives_irrelevant_chunk_a_low_score(
    real_client: EmbeddingClient,
) -> None:
    """无关 chunk 的分数应该很低——这是 Task 8 能定相似度阈值的前提。

    实测（multi-qa-MiniLM-L6-cos-v1）：无关 chunk 落在 0.03~0.11；
    旧模型 all-MiniLM-L6-v2 的地板高得多（0.11~0.19）。底噪低，
    "多低算不相关"这条线才画得出来。分数上限用 0.3 留足余量，
    这个断言要防的是"某个模型对所有输入都给高分"这种退化情况。
    """
    question = real_client.embed_text("How does the model handle long-range dependencies?")
    unrelated = real_client.embed_text(
        "We trained on the standard WMT 2014 English-German dataset consisting of about "
        "4.5 million sentence pairs, encoded using byte-pair encoding."
    )

    assert _cosine(question, unrelated) < 0.3


def test_known_limitation_single_sentence_with_keyword_sharing_distractor(
    real_client: EmbeddingClient,
) -> None:
    """已知局限：候选段落都是**单句**、且干扰句恰好含问题里的常见词时，会挑错。

    实测（2026-09-18，multi-qa-MiniLM-L6-cos-v1）:

        问句   "How does the model handle long-range dependencies?"
        相关句 "Self-attention relates all positions in a sequence ..."  → 0.173
        干扰句 "We trained the model for three days on eight GPUs."     → 0.245  ← 更高

    拆解出来的原因（对照实验）：

        相关句 vs 干扰句**不含** model → 0.173 vs 0.084  ✅ 命中
        相关句 vs 干扰句**含**   model → 0.173 vs 0.245  ❌ 挑错

    也就是说：相关句和问句**一个实词都不共用**，而干扰句共用了一个 "model"。
    单句词汇面太窄，"一个共享词"就压过了语义。

    **这是词汇驱动的检索在 384 维小模型上的固有边界**，不是某个模型的 bug：
    换 all-MiniLM-L6-v2 同样失败（0.169 vs 0.308）。上面那个 chunk 级测试证明
    真实系统不受影响——所以这个测试的作用是**标出边界在哪**，而不是报警。

    它同时是 Task 14 离线评估的基线：如果评估里出现大量"问题和所有候选都不共用词"
    的用例，命中率就会掉，那是数据特性而非实现缺陷。
    """
    question = real_client.embed_text("How does the model handle long-range dependencies?")
    related_but_no_shared_content_word = real_client.embed_text(
        "Self-attention relates all positions in a sequence to compute its representation."
    )
    unrelated_but_shares_the_word_model = real_client.embed_text(
        "We trained the model for three days on eight GPUs."
    )

    assert _cosine(question, unrelated_but_shares_the_word_model) > _cosine(
        question, related_but_no_shared_content_word
    )
