"""HTTP 层的输入/输出结构。

这些类和数据库模型（app/models/tables.py）是**两套东西**，故意不共用：

- 数据库模型描述"数据怎么存"——有主键、外键、约束、级联。
- schema 描述"接口怎么收发"——只有调用方需要看到的字段。

耦合它们看起来省事，但会立刻带来两个问题：
1. 加一个内部字段（比如 embedding）就会自动暴露给客户端；
2. 改表结构会意外改掉接口契约，客户端无声无息地崩。

所以边界处要显式转换：请求 → schema → service → ORM → schema → 响应。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreateRequest(BaseModel):
    """POST /documents 的请求体。

    这里的 Field 约束是**第一道**校验：类型不对、字段缺失、长度越界，
    在进入业务代码之前就被 pydantic 拦下，自动返回 422。
    它只能表达"形状"层面的规则（长度、类型），表达不了"标题不能全是空格"
    这种业务规则——那种校验在 service 层。
    """

    title: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)


class DocumentCreateResponse(BaseModel):
    document_id: UUID
    chunk_count: int
    # 同一篇内容重复导入时 created=False。让调用方能区分
    # "这次真的入库了" 和 "这篇之前就导过了"。
    created: bool
    # 这次请求里**新写入向量**的片段数。
    #
    # 为什么要暴露这个数字：导入和索引是两步，如果只回报 chunk_count，
    # 调用方就看不出索引到底有没有发生——一个"文档存进去了但全是 NULL 向量"
    # 的结果，和完全成功长得一模一样。这正是 F-39 那个缺口能藏那么久的原因。
    #
    # 三种取值分别意味着：
    #   = chunk_count → 全新文档，全部片段都编码了
    #   = 0 且 created=False → 重复导入，且这篇早就索引过了
    #   > 0 且 created=False → 重复导入，但补上了之前缺的向量（自愈）
    embedded_chunk_count: int


class DocumentListItem(BaseModel):
    """GET /documents 列表里的每一项。

    from_attributes=True 允许 pydantic 直接读 ORM 对象的属性
    （document.title 这样取），而不用先手工转成 dict。
    没有它，model_validate(orm_object) 会报错，因为 pydantic 默认只认字典。

    注意这里**不含 content_hash**：它是内部去重用的指纹，
    对客户端没有意义，暴露出去反而会让人以为可以拿它做什么。
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    source: str
    created_at: datetime
