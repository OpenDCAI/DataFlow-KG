import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class FinKGRelationExtractorPrompt(PromptABC):
    """
    从金融文本中抽取带时间的关系四元组：

    <subj> Entity <obj> Entity <rel> Relation <time> TimeValue

    同时输出 entity_class:
    [HeadEntityClass, TailEntityClass]

    entity_class 必须是 ontology 中的最小类别（leaf type）
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):

        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        relation_str = ", ".join(relation_list)

        if self.lang == "en":

            self.system_text = textwrap.dedent(f"""
You are an expert in extracting temporal knowledge graph quadruples from financial text.

You are given a predefined ontology specifying valid entity types and relations.

Each quadruple MUST follow this format:

<subj> Entity <obj> Entity <rel> Relation <time> TimeValue


=== ENTITY RULES ===

Entities MUST belong to the following ontology leaf types ONLY:

{entity_str}

Rules:
- Use only entities appearing in the text
- Do NOT use pronouns
- Do NOT invent entities
- Do NOT use high-level ontology categories


=== RELATION RULES ===

Relations MUST be one of the following:

{relation_str}

Rules:
- Do NOT invent relations
- Do NOT use high-level relation categories


=== ENTITY CLASS RULES (IMPORTANT) ===

For each tuple you MUST output the entity classes.

Rules:

- Entity classes MUST be the most specific ontology types (leaf types)
- Do NOT output high-level ontology categories
- Classes MUST correspond to the entities in the tuple
- Class order MUST match entity order

Example:

Tuple:
<subj> Goldman Sachs <obj> IssuerRole <rel> plays_role <time> 2025

Entity classes:
["Corporation","Corporation"]


CRITICAL CONSTRAINT:

If the correct leaf class cannot be determined from the ontology,
DO NOT output the tuple.


=== TIME STANDARDIZATION ===

Use the following formats:

Specific date:
YYYY-MM-DD

Month:
Month YYYY
Example: March 2025

Year:
YYYY

Quarter:
QX YYYY
Example: Q3 2025

Time interval:
YYYY-MM-DD|YYYY-MM-DD

If no time is mentioned:
Use NA


=== OUTPUT FORMAT ===

Return JSON ONLY.

{{
  "tuple":[
    "<subj> Entity <obj> Entity <rel> Relation <time> TimeValue"
  ],
  "entity_class":[
    ["HeadEntityClass","TailEntityClass"]
  ]
}}

Do NOT output explanations.
""")

        else:

            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱关系抽取专家。

需要从文本中抽取关系四元组：

<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值


=== 实体规则 ===

实体必须属于以下本体**最底层类别**：

{entity_str}

规则：

- 仅使用文本中出现的实体
- 禁止使用代词
- 不得虚构实体
- 不得使用高层本体类别


=== 关系规则 ===

关系必须属于以下底层关系：

{relation_str}

规则：

- 不得虚构关系
- 不得使用高层关系类别


=== 实体类别规则（重要） ===

每个四元组必须同时输出实体类别。

规则：

- 实体类别必须是本体中的**最小类别（leaf class）**
- 不允许使用高层类别
- 类别顺序必须与实体顺序一致

示例：

四元组：

<subj> 高盛 <obj> IssuerRole <rel> plays_role <time> 2025

实体类别：

["Corporation","Corporation"]


强约束：

如果无法确定实体对应的最小类别，
则不要输出该四元组。


=== 时间标准化 ===

具体日期：
YYYY-MM-DD

月份：
Month YYYY

年份：
YYYY

季度：
QX YYYY
示例：Q3 2025

时间区间：
YYYY-MM-DD|YYYY-MM-DD

如果没有时间：
使用 NA


=== 输出格式 ===

仅输出 JSON：

{{
  "tuple":[
    "<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值"
  ],
  "entity_class":[
    ["头实体类别","尾实体类别"]
  ]
}}

不要输出解释文本。
""")

        return self.system_text

    def build_prompt(self, text: str):

        if self.lang == "en":

            return textwrap.dedent(f"""
Extract temporal financial relation quadruples from the text.

Use ONLY ontology entities and relations.

Text:
{text}

Output JSON ONLY:

{{
  "tuple":[
    "<subj> Entity <obj> Entity <rel> Relation <time> TimeValue"
  ],
  "entity_class":[
    ["HeadEntityClass","TailEntityClass"]
  ]
}}
""")

        else:

            return textwrap.dedent(f"""
从以下文本中抽取金融关系四元组。

文本：
{text}

仅输出 JSON：

{{
  "tuple":[
    "<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值"
  ],
  "entity_class":[
    ["实体类别","实体类别"]
  ]
}}
""")


@PROMPT_REGISTRY.register()
class FinKGAttributeExtractorPrompt(PromptABC):
    """
    从金融文本中抽取属性四元组：

    <subj> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue

    同时输出 entity_class:
    [EntityClass]
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):

        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        attribute_list = []
        for group in ontology.get("attribute_type", {}).values():
            attribute_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        attribute_str = ", ".join(attribute_list)

        if self.lang == "en":

            self.system_text = textwrap.dedent(f"""
You are an expert in extracting temporal attribute quadruples from financial text.

Each quadruple MUST follow this format:

<subj> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue


=== ENTITY RULES ===

Entities MUST belong to:

{entity_str}


=== ATTRIBUTE RULES ===

Attributes MUST belong to:

{attribute_str}


=== ENTITY CLASS RULES ===

Each tuple MUST also output the entity class.

Rules:

- Entity class MUST be the most specific ontology type (leaf type)
- Do NOT output high-level ontology categories
- The class must correspond to the entity


Example:

Tuple:

<subj> Apple <attribute> market_cap <value> 3.1T USD <time> Q3 2025

Entity class:

["Corporation"]


CRITICAL CONSTRAINT:

If the correct leaf class cannot be determined,
DO NOT output the tuple.


=== TIME STANDARDIZATION ===

YYYY-MM-DD

Month YYYY

YYYY

QX YYYY

YYYY-MM-DD|YYYY-MM-DD

If no time:
NA


=== OUTPUT FORMAT ===

Return JSON ONLY:

{{
  "tuple":[
    "<subj> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue"
  ],
  "entity_class":[
    ["EntityClass"]
  ]
}}
""")

        else:

            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱属性抽取专家。

需要抽取属性四元组：

<subj> 实体 <attribute> 属性 <value> 属性值 <time> 时间值


=== 实体规则 ===

实体必须属于以下最小类别：

{entity_str}


=== 属性规则 ===

属性必须属于：

{attribute_str}


=== 实体类别规则 ===

每个四元组必须输出实体类别。

规则：

- 实体类别必须是本体最小类别（leaf class）
- 不允许使用高层类别


示例：

<subj> 苹果公司 <attribute> market_cap <value> 3.1T USD <time> Q3 2025

实体类别：

["Corporation"]


强约束：

如果无法确定实体对应的最小类别，
则不要输出该四元组。


=== 时间标准化 ===

YYYY-MM-DD
Month YYYY
YYYY
QX YYYY
YYYY-MM-DD|YYYY-MM-DD
NA


=== 输出格式 ===

仅输出 JSON：

{{
  "tuple":[
    "<subj> 实体 <attribute> 属性 <value> 属性值 <time> 时间值"
  ],
  "entity_class":[
    ["实体类别"]
  ]
}}
""")

        return self.system_text

    def build_prompt(self, text: str):

        if self.lang == "en":

            return textwrap.dedent(f"""
Extract temporal financial attribute quadruples.

Text:
{text}

Output JSON ONLY:

{{
  "tuple":[
    "<subj> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue"
  ],
  "entity_class":[
    ["EntityClass"]
  ]
}}
""")

        else:

            return textwrap.dedent(f"""
从以下文本中抽取金融属性四元组。

文本：
{text}

仅输出 JSON：

{{
  "tuple":[
    "<subj> 实体 <attribute> 属性 <value> 属性值 <time> 时间值"
  ],
  "entity_class":[
    ["实体类别"]
  ]
}}
""")


@PROMPT_REGISTRY.register()
class FinKGTableSchemaPrompt(PromptABC):
    """
    理解任意金融表格的结构语义，为后续表格到知识图谱抽取提供 schema。
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):

        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)

        attribute_list = []
        for group in ontology.get("attribute_type", {}).values():
            attribute_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        relation_str = ", ".join(relation_list)
        attribute_str = ", ".join(attribute_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""
You are an expert in understanding arbitrary financial tables for knowledge graph construction.

You are given a financial table that may be written as markdown, CSV-like text, or JSON records.
Your task is to infer the table schema needed for downstream KG extraction.

Use ONLY the following ontology leaf types when proposing candidate entity types:
{entity_str}

Use ONLY the following ontology leaf relations when proposing candidate relations:
{relation_str}

Use ONLY the following ontology leaf attributes when proposing candidate attributes:
{attribute_str}

Schema rules:
- Focus on how one row should be interpreted
- Identify which columns contain entities, time, relation cues, and attribute values
- Do NOT extract tuples yet
- If the table is mostly about institution profile / metadata, prefer attributes
- If the table is mostly about parent-child / ownership / control links, prefer relations
- If a relation is not explicit, leave candidate_relations empty
- If an attribute is not explicit, leave candidate_attributes empty

Return JSON ONLY in this format:
{{
  "table_type": "short schema label",
  "primary_entity_columns": ["col_a"],
  "secondary_entity_columns": ["col_b"],
  "time_columns": ["col_time"],
  "relation_columns": ["col_rel_hint"],
  "attribute_columns": ["col_attr"],
  "value_columns": ["col_value"],
  "candidate_entity_types": {{
    "col_a": ["Corporation"],
    "col_b": ["Corporation"]
  }},
  "candidate_relations": ["owns"],
  "candidate_attributes": ["legal_name"],
  "row_semantics": "One row means ..."
}}
""")
        else:
            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱表格理解专家。

你将看到一张金融表格，输入可能是 markdown 表格、类似 CSV 的文本，或者 JSON records。
你的任务是先理解这张表格的 schema，为后续知识图谱抽取提供结构化语义。

候选实体类别只能来自以下本体叶子类型：
{entity_str}

候选关系只能来自以下本体叶子关系：
{relation_str}

候选属性只能来自以下本体叶子属性：
{attribute_str}

规则：
- 重点判断“一行代表什么”
- 识别实体列、时间列、关系线索列、属性列、数值列
- 这一阶段不要直接抽取四元组
- 如果表格主要描述机构画像或元数据，优先考虑属性型 schema
- 如果表格主要描述 parent-child / ownership / control，优先考虑关系型 schema
- 如果关系不明确，candidate_relations 返回空列表
- 如果属性不明确，candidate_attributes 返回空列表

仅输出 JSON，格式如下：
{{
  "table_type": "简短 schema 标签",
  "primary_entity_columns": ["列A"],
  "secondary_entity_columns": ["列B"],
  "time_columns": ["时间列"],
  "relation_columns": ["关系线索列"],
  "attribute_columns": ["属性列"],
  "value_columns": ["取值列"],
  "candidate_entity_types": {{
    "列A": ["Corporation"],
    "列B": ["Corporation"]
  }},
  "candidate_relations": ["owns"],
  "candidate_attributes": ["legal_name"],
  "row_semantics": "一行表示什么"
}}
""")

        return self.system_text

    def build_prompt(
        self,
        table_text: str,
        table_title: str = "",
        table_context: str = "",
    ):
        if self.lang == "en":
            return textwrap.dedent(f"""
Infer the table schema for downstream financial KG extraction.

Table title:
{table_title or "NA"}

Table context:
{table_context or "NA"}

Table content:
{table_text}

Return JSON ONLY.
""")

        return textwrap.dedent(f"""
请为后续金融知识图谱抽取推断这张表格的 schema。

表格标题：
{table_title or "NA"}

表格上下文：
{table_context or "NA"}

表格内容：
{table_text}

仅输出 JSON。
""")


@PROMPT_REGISTRY.register()
class FinKGTableTupleExtractionPrompt(PromptABC):
    """
    基于表格 schema 与表格内容，抽取金融知识图谱四元组。
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):

        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)

        attribute_list = []
        for group in ontology.get("attribute_type", {}).values():
            attribute_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        relation_str = ", ".join(relation_list)
        attribute_str = ", ".join(attribute_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""
You are an expert in converting arbitrary financial tables into Financial KG quadruples.

You must extract tuples using ONLY the ontology leaf types below.

Valid entity types:
{entity_str}

Valid relations:
{relation_str}

Valid attributes:
{attribute_str}

You may output two tuple formats:

Relation tuple:
<subj> Entity <obj> Entity <rel> Relation <time> TimeValue

Attribute tuple:
<entity> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue

Important rules:
- Extract only facts directly supported by the table
- Do NOT invent entities, relations, attributes, or values
- Prefer the most semantically specific relation explicitly supported by the table
- If the table shows a parent-child hierarchy, prefer parent_of / subsidiary_of
- If the table shows direct equity holding, prefer owns
- If the table shows control without equity evidence, prefer controls
- For institution profile rows, prefer attribute tuples
- Use the most specific ontology leaf type for entity_class
- For relation tuples, entity_class must be ["HeadEntityClass", "TailEntityClass"]
- For attribute tuples, entity_class must be ["EntityClass"]
- The number and order of entity_class entries must match tuple order

Time standardization:
- Specific date: YYYY-MM-DD
- Month: Month YYYY
- Year: YYYY
- Quarter: QX YYYY
- Interval: YYYY-MM-DD|YYYY-MM-DD
- If unavailable: NA

Return JSON ONLY:
{{
  "tuple": [
    "<subj> Entity <obj> Entity <rel> Relation <time> TimeValue",
    "<entity> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue"
  ],
  "entity_class": [
    ["HeadEntityClass", "TailEntityClass"],
    ["EntityClass"]
  ]
}}
""")
        else:
            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱表格转四元组专家。

你必须只使用以下本体叶子类型进行抽取。

合法实体类型：
{entity_str}

合法关系：
{relation_str}

合法属性：
{attribute_str}

你可以输出两种四元组：

关系型四元组：
<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值

属性型四元组：
<entity> 实体 <attribute> 属性 <value> 属性值 <time> 时间值

重要规则：
- 只抽取表格直接支持的事实
- 不得虚构实体、关系、属性或取值
- 优先选择表格明确支持的最具体关系
- 如果表格表达 parent-child 层级，优先使用 parent_of / subsidiary_of
- 如果表格表达直接股权持有，优先使用 owns
- 如果表格表达控制但没有股权证据，优先使用 controls
- 如果表格主要是机构画像，优先输出属性型四元组
- entity_class 必须是最小叶子类型
- 对关系型四元组，entity_class 格式必须是 ["头实体类别", "尾实体类别"]
- 对属性型四元组，entity_class 格式必须是 ["实体类别"]
- entity_class 的数量和顺序必须与 tuple 严格对应

时间标准化：
- 具体日期：YYYY-MM-DD
- 月份：Month YYYY
- 年份：YYYY
- 季度：QX YYYY
- 区间：YYYY-MM-DD|YYYY-MM-DD
- 缺失：NA

仅输出 JSON：
{{
  "tuple": [
    "<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值",
    "<entity> 实体 <attribute> 属性 <value> 属性值 <time> 时间值"
  ],
  "entity_class": [
    ["头实体类别", "尾实体类别"],
    ["实体类别"]
  ]
}}
""")

        return self.system_text

    def build_prompt(
        self,
        table_text: str,
        schema_json: str,
        table_title: str = "",
        table_context: str = "",
    ):
        if self.lang == "en":
            return textwrap.dedent(f"""
Convert the financial table into Financial KG quadruples using the inferred schema.

Table title:
{table_title or "NA"}

Table context:
{table_context or "NA"}

Inferred schema:
{schema_json}

Table content:
{table_text}

Return JSON ONLY.
""")

        return textwrap.dedent(f"""
请基于推断出的 schema，将这张金融表格转换为金融知识图谱四元组。

表格标题：
{table_title or "NA"}

表格上下文：
{table_context or "NA"}

推断出的 schema：
{schema_json}

表格内容：
{table_text}

仅输出 JSON。
""")


@PROMPT_REGISTRY.register()
class FinKGInvestmentAnalysisPrompt(PromptABC):
    """
    基于金融知识图谱上下文生成投资分析结论。
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)
        relation_str = ", ".join(relation_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""
You are a financial knowledge graph investment analysis expert.

You are given:
1. A target investable entity
2. Evidence tuples extracted from a Financial KG
3. Optional recent external market news context

Available Financial KG relations:
{relation_str}

Task:
- Assess whether the target entity appears structurally attractive or risky from an investment-research perspective
- Focus on ownership/control structure, financing dependencies, guarantee links,
  regulatory pressure, counterparty exposure, and event-sensitive signals
- Use the Financial KG for structural evidence and the recent market news for timely catalysts
- Do NOT give direct buy/sell advice
- Do NOT invent facts beyond the tuples and the provided news context

Output rules:
- analysis_summary: one concise paragraph
- bullish_signals: positive or stabilizing evidence-backed signals
- bearish_signals: negative or adverse evidence-backed signals
- watch_items: unresolved items worth monitoring
- key_paths: supporting evidence paths built ONLY by copying exact tuple strings from the input evidence and joining them with " || "
- confidence: one of high / medium / low

Path constraints:
- Every path item must be composed of one or more exact evidence tuples
- NEVER paraphrase tuple content inside key_paths
- NEVER reverse tuple direction
- If no valid path can be formed, return an empty list for key_paths

Return JSON ONLY:
{{
  "analysis_summary": "string",
  "bullish_signals": ["signal 1"],
  "bearish_signals": ["signal 1"],
  "watch_items": ["item 1"],
  "key_paths": ["tuple1 || tuple2"],
  "confidence": "medium"
}}
""")
        else:
            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱投资分析专家。

你将获得：
1. 目标投资分析实体
2. 来自金融知识图谱的证据四元组
3. 可选的近期外部市场新闻上下文

可用金融关系如下：
{relation_str}

任务要求：
- 从投资研究视角评估目标实体是否具备结构性吸引力或潜在风险
- 重点关注股权/控制结构、融资依赖、担保链、监管压力、交易对手暴露、事件敏感信号
- 以金融知识图谱提供结构性证据，以近期市场新闻提供时效性催化信息
- 不要直接给出买入/卖出建议
- 不得虚构超出四元组和新闻上下文范围的事实

输出规则：
- analysis_summary：一段简洁总结
- bullish_signals：有利或稳定性信号
- bearish_signals：不利或负面信号
- watch_items：值得持续跟踪的问题
- key_paths：支撑分析的关键路径，只能由输入证据中的原始四元组逐字复制并用 " || " 连接
- confidence：high / medium / low 之一

路径约束：
- 每个 path 必须由一个或多个输入证据四元组原文组成
- 在 key_paths 中严禁改写或总结四元组内容
- 严禁颠倒四元组方向
- 如果无法形成合法路径，则 key_paths 返回空列表

仅输出 JSON：
{{
  "analysis_summary": "字符串",
  "bullish_signals": ["信号1"],
  "bearish_signals": ["信号1"],
  "watch_items": ["事项1"],
  "key_paths": ["四元组1 || 四元组2"],
  "confidence": "medium"
}}
""")

        return self.system_text

    def build_prompt(
        self,
        target_entity: str,
        tuple_text: str,
        market_news_context: str = "",
    ):
        if self.lang == "en":
            return textwrap.dedent(f"""
Assess the target entity from an investment perspective using the Financial KG evidence.

Target entity:
{target_entity or "NA"}

Evidence tuples:
{tuple_text}

Recent market news context:
{market_news_context or "NA"}

Return JSON ONLY.
""")

        return textwrap.dedent(f"""
请基于金融知识图谱证据，从投资分析角度评估目标实体。

目标实体：
{target_entity or "NA"}

证据四元组：
{tuple_text}

近期市场新闻上下文：
{market_news_context or "NA"}

仅输出 JSON。
""")


@PROMPT_REGISTRY.register()
class FinKGEventQueryExtractionPrompt(PromptABC):
    """
    从金融事件文本中抽取事件摘要和锚定实体。
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)
        entity_str = ", ".join(entity_leaf_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""
You are an expert in understanding financial event text for knowledge-graph tracing.

Read the event text and extract:
1. target_event: a short normalized event summary
2. anchor_entities: explicit entities mentioned in the text that are useful for tracing impact paths

Available entity leaf types:
{entity_str}

Rules:
- Extract only entities explicitly mentioned in the text
- Prefer institutions, instruments, regulators, agreements, and named events
- Do NOT invent entities not grounded in the text
- Do NOT output generic type labels such as CorporateBond or RegulatoryAction
- Do NOT output pure years or dates as entities unless they are part of a named event
- target_event should be a short phrase, not a full sentence
- anchor_entities should be a JSON array of strings

Return JSON ONLY:
{{
  "target_event": "short event summary",
  "anchor_entities": ["entity 1", "entity 2"]
}}
""")
        else:
            self.system_text = textwrap.dedent(f"""
你是一名金融事件文本理解专家。

请阅读事件文本，并抽取：
1. target_event：简短规范化事件摘要
2. anchor_entities：文本中显式出现、适合用于图谱回溯的锚定实体

可用实体叶子类型：
{entity_str}

规则：
- 只能抽取文本中明确出现的实体
- 优先抽机构、金融工具、监管方、协议、命名事件
- 不得虚构文本中不存在的实体
- 不要输出 CorporateBond、RegulatoryAction 这类泛化类型标签
- 不要把纯年份或日期当作实体，除非它是命名事件的一部分
- target_event 应是简短短语，不要写完整句子
- anchor_entities 必须是字符串数组

仅输出 JSON：
{{
  "target_event": "简短事件摘要",
  "anchor_entities": ["实体1", "实体2"]
}}
""")

        return self.system_text

    def build_prompt(self, raw_event_text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""
Extract the target event and anchor entities from the following financial event text.

Event text:
{raw_event_text}

Return JSON ONLY.
""")

        return textwrap.dedent(f"""
请从以下金融事件文本中抽取目标事件和锚定实体。

事件文本：
{raw_event_text}

仅输出 JSON。
""")


@PROMPT_REGISTRY.register()
class FinKGEventImpactTracingPrompt(PromptABC):
    """
    基于金融知识图谱上下文追踪事件影响路径。
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)
        relation_str = ", ".join(relation_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""
You are a financial knowledge graph event impact tracing expert.

You are given:
1. A target financial event or event query
2. Optional anchor entities extracted from user input
3. Optional original event text from the user
4. Evidence tuples from a Financial KG

Available Financial KG relations:
{relation_str}

Task:
- Trace which entities are most likely impacted by the event
- Emphasize propagation through ownership, guarantee, financing, default,
  regulatory, and occurrence relations
- Use ONLY the provided tuples as evidence
- Do NOT invent entities or impact paths

Output rules:
- analysis_summary: concise event impact summary
- impacted_entities: entities directly or indirectly affected
- impact_types: short labels such as regulatory, credit, funding, governance, contagion
- impact_paths: propagation paths built ONLY by copying exact tuple strings from the input evidence and joining them with " || "
- confidence: one of high / medium / low

Path constraints:
- Every path item must be composed of one or more exact evidence tuples
- NEVER paraphrase tuple content inside impact_paths
- NEVER reverse tuple direction
- If no valid path can be formed, return an empty list for impact_paths

Return JSON ONLY:
{{
  "analysis_summary": "string",
  "impacted_entities": ["entity 1"],
  "impact_types": ["regulatory"],
  "impact_paths": ["tuple1 || tuple2"],
  "confidence": "medium"
}}
""")
        else:
            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱事件影响追踪专家。

你将获得：
1. 目标金融事件或事件问题
2. 可选的锚定实体
3. 可选的原始事件文本
4. 来自金融知识图谱的证据四元组

可用金融关系如下：
{relation_str}

任务要求：
- 追踪哪些实体最可能受到该事件影响
- 重点关注股权、担保、融资、违约、监管、事件关系带来的影响传导
- 只能依据提供的四元组进行判断
- 不得虚构实体或影响路径

输出规则：
- analysis_summary：事件影响总结
- impacted_entities：直接或间接受影响的实体
- impact_types：简短标签，如 regulatory、credit、funding、governance、contagion
- impact_paths：展示影响传导的路径字符串，只能由输入证据中的原始四元组逐字复制并用 " || " 连接
- confidence：high / medium / low 之一

路径约束：
- 每个 path 必须由一个或多个输入证据四元组原文组成
- 在 impact_paths 中严禁改写或总结四元组内容
- 严禁颠倒四元组方向
- 如果无法形成合法路径，则 impact_paths 返回空列表

仅输出 JSON：
{{
  "analysis_summary": "字符串",
  "impacted_entities": ["实体1"],
  "impact_types": ["regulatory"],
  "impact_paths": ["四元组1 || 四元组2"],
  "confidence": "medium"
}}
""")

        return self.system_text

    def build_prompt(
        self,
        target_event: str,
        target_entity: str,
        tuple_text: str,
        raw_event_text: str = "",
    ):
        if self.lang == "en":
            return textwrap.dedent(f"""
Trace the event impact using the Financial KG evidence.

Target event:
{target_event or "NA"}

Anchor entity:
{target_entity or "NA"}

Original event text:
{raw_event_text or "NA"}

Evidence tuples:
{tuple_text}

Return JSON ONLY.
""")

        return textwrap.dedent(f"""
请基于金融知识图谱证据，追踪该事件的影响路径。

目标事件：
{target_event or "NA"}

锚定实体：
{target_entity or "NA"}

原始事件文本：
{raw_event_text or "NA"}

证据四元组：
{tuple_text}

仅输出 JSON。
""")


@PROMPT_REGISTRY.register()
class FinKGEntityRiskAssessmentPrompt(PromptABC):
    """
    基于金融知识图谱上下文进行实体风险预估。
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)
        relation_str = ", ".join(relation_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""
You are a financial knowledge graph risk assessment expert.

You are given:
1. A target entity
2. Evidence tuples from a Financial KG

Available Financial KG relations:
{relation_str}

Task:
- Estimate the target entity's overall risk profile
- Consider guarantee chains, default links, financing/counterparty dependencies,
  regulatory actions, ownership/control concentration, and contagion paths
- Treat the analysis objective itself as an implicit built-in query:
  infer how risk can propagate to, concentrate on, or structurally affect the target entity
- Use ONLY the provided tuples as evidence
- Do NOT invent risks unsupported by the tuples

Output rules:
- analysis_summary: concise risk assessment
- risk_types: short labels such as credit_risk, regulatory_risk, contagion_risk
- risk_entities: entities through which risk is transmitted, concentrated, or surfaced
- risk_paths: key risk propagation paths built ONLY by copying exact tuple strings from the input evidence and joining them with " || "
- risk_score: an integer from 0 to 100, where higher means higher estimated overall risk

Path constraints:
- Every path item must be composed of one or more exact evidence tuples
- NEVER paraphrase tuple content inside risk_paths
- NEVER reverse tuple direction
- If no valid path can be formed, return an empty list for risk_paths

Return JSON ONLY:
{{
  "analysis_summary": "string",
  "risk_types": ["credit_risk"],
  "risk_entities": ["entity 1"],
  "risk_paths": ["tuple1 || tuple2"],
  "risk_score": 72
}}
""")
        else:
            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱风险预估专家。

你将获得：
1. 目标实体
2. 来自金融知识图谱的证据四元组

可用金融关系如下：
{relation_str}

任务要求：
- 评估目标实体的整体风险画像
- 重点考虑担保链、违约链、融资/交易对手依赖、监管动作、股权/控制集中度、风险传染路径
- 将分析目标视为内置问题：
  推断风险如何通过现有关系传导至目标实体、在目标实体上集中，或结构性影响目标实体
- 只能依据提供的四元组进行分析
- 不得虚构没有证据支持的风险

输出规则：
- analysis_summary：简洁风险预估总结
- risk_types：简短标签，如 credit_risk、regulatory_risk、contagion_risk
- risk_entities：体现风险传导、集中或显化的关键实体
- risk_paths：展示关键风险传导路径，只能由输入证据中的原始四元组逐字复制并用 " || " 连接
- risk_score：0 到 100 的整数，分数越高表示整体预估风险越高

路径约束：
- 每个 path 必须由一个或多个输入证据四元组原文组成
- 在 risk_paths 中严禁改写或总结四元组内容
- 严禁颠倒四元组方向
- 如果无法形成合法路径，则 risk_paths 返回空列表

仅输出 JSON：
{{
  "analysis_summary": "字符串",
  "risk_types": ["credit_risk"],
  "risk_entities": ["实体1"],
  "risk_paths": ["四元组1 || 四元组2"],
  "risk_score": 72
}}
""")

        return self.system_text

    def build_prompt(
        self,
        target_entity: str,
        tuple_text: str,
    ):
        if self.lang == "en":
            return textwrap.dedent(f"""
Estimate the target entity's risk profile using the Financial KG evidence.

Target entity:
{target_entity or "NA"}

Evidence tuples:
{tuple_text}

Return JSON ONLY.
""")

        return textwrap.dedent(f"""
请基于金融知识图谱证据，对目标实体进行风险预估。

目标实体：
{target_entity or "NA"}

证据四元组：
{tuple_text}

仅输出 JSON。
""")


FinKGEntityRiskExposureAnalysisPrompt = FinKGEntityRiskAssessmentPrompt
