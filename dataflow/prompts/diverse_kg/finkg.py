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
class FinKGRelationChainInferencePrompt(PromptABC):
    """
    金融KG多跳关系链推理 Prompt。

    输入：目标实体对 + 相关四元组（k-hop 邻域） + 本体关系列表
    输出：推断的关系四元组 + 实体类别
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
You are an expert in Financial KG multi-hop relation reasoning.

You are given:
1. A target entity pair
2. Related relation quadruples (k-hop neighborhood)
3. Financial ontology relation list

Quadruple format:
<subj> Entity <obj> Entity <rel> Relation <time> TimeValue


=== AVAILABLE RELATIONS ===

{relation_str}


=== TASK ===

Infer new relation quadruple(s) between the target entity pair.

Rules:
- Use only evidence-supported reasoning from provided tuples
- Do NOT invent entities
- Inferred relations MUST come from the ontology list above
- If evidence is insufficient, output empty lists


=== REASONING EXAMPLES (guidance, NOT fixed rules) ===

The examples below use relations from the ontology to illustrate multi-hop patterns.

Example A — Ownership penetration:
A owns B
B owns C
=> A controls C

Example B — Guarantee chain propagation:
A guarantor_of B
B guarantor_of C
=> A guarantor_of C

Example C — Related-party detection:
A subsidiary_of X
B subsidiary_of X
=> A affects B

Example D — Cross-relation impact:
A lends_to B
B defaults_on C
=> A affects C


=== TIME RULE ===

Infer the most reasonable time from evidence.
If uncertain, use NA.


=== ENTITY CLASS RULES ===

For each inferred tuple, output the entity classes.
- Classes must be ontology leaf types
- Order must match entity order in the tuple


=== OUTPUT FORMAT ===

Return JSON ONLY:

{{
  "tuple": [
    "<subj> Entity <obj> Entity <rel> Relation <time> TimeValue"
  ],
  "entity_class": [
    ["HeadEntityClass", "TailEntityClass"]
  ]
}}

Do NOT output explanations.
""")

        else:

            self.system_text = textwrap.dedent(f"""
你是一名金融知识图谱多跳关系推理专家。

你将获得：
1. 目标实体对
2. 相关关系四元组（k-hop 邻域）
3. 金融本体关系列表

四元组格式：
<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值


=== 可用关系 ===

{relation_str}


=== 任务 ===

推断目标实体对之间的新关系四元组。

规则：
- 推断必须有输入四元组证据支持
- 不得虚构实体
- 推断的关系必须来自上方本体关系列表
- 若证据不足，输出空列表


=== 推理示例（仅作引导，不是固定规则） ===

以下示例使用本体中的关系来展示多跳推理模式。

示例A — 股权穿透：
A owns B
B owns C
=> A controls C

示例B — 担保链传导：
A guarantor_of B
B guarantor_of C
=> A guarantor_of C

示例C — 关联方识别：
A subsidiary_of X
B subsidiary_of X
=> A affects B

示例D — 跨关系影响传导：
A lends_to B
B defaults_on C
=> A affects C


=== 时间规则 ===

尽量根据证据给出合理时间。
无法判断时使用 NA。


=== 实体类别规则 ===

每个推断四元组必须输出实体类别。
- 类别必须是本体最小类别（leaf type）
- 顺序必须与四元组中的实体顺序一致


=== 输出格式 ===

仅输出 JSON：

{{
  "tuple": [
    "<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值"
  ],
  "entity_class": [
    ["头实体类别", "尾实体类别"]
  ]
}}

不要输出解释文本。
""")

        return self.system_text

    def build_prompt(
        self,
        entity1: str,
        entity2: str,
        tuples: list
    ):
        tuple_text = "\n".join(tuples)

        if self.lang == "en":

            return textwrap.dedent(f"""
Infer relation quadruple(s) between the target entities using the related tuples.

Target Entity 1:
{entity1}

Target Entity 2:
{entity2}

Related quadruples:
{tuple_text}

Requirements:
- Infer only relations between the target entity pair
- Infer only if evidence is sufficient
- Output JSON only

Output JSON ONLY:

{{
  "tuple": [
    "<subj> Entity <obj> Entity <rel> Relation <time> TimeValue"
  ],
  "entity_class": [
    ["HeadEntityClass", "TailEntityClass"]
  ]
}}
""")

        else:

            return textwrap.dedent(f"""
请基于相关四元组，推断目标实体对之间的新关系四元组。

目标实体1：
{entity1}

目标实体2：
{entity2}

相关四元组：
{tuple_text}

要求：
- 只推断目标实体对之间的关系
- 证据不足时不输出推断
- 严格输出 JSON

仅输出 JSON：

{{
  "tuple": [
    "<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值"
  ],
  "entity_class": [
    ["头实体类别", "尾实体类别"]
  ]
}}
""")
