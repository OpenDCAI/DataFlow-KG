import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class GeoKGRelationExtractorPrompt(PromptABC):
    """
    从地理文本中抽取带时间的关系四元组：

    <subj> Entity <obj> Entity <rel> Relation <time> TimeValue

    同时输出 entity_class:
    [HeadEntityClass, TailEntityClass]

    entity_class 必须是 ontology 中的最小类别（leaf type）
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None


    def build_system_prompt(self, ontology: dict):

        # leaf entity types
        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        # relation types
        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        relation_str = ", ".join(relation_list)

        if self.lang == "en":

            self.system_text = textwrap.dedent(f"""
You are an expert in extracting spatio-temporal knowledge graph quadruples from geographic text.

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
<subj> Mount Fuji <obj> Japan <rel> located_in <time> NA

Entity classes:
["Mountain","Country"]


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
Q1 YYYY

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
你是一名地理时空知识图谱关系抽取专家。

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

<subj> 富士山 <obj> 日本 <rel> 位于 <time> NA

实体类别：

["Mountain","Country"]


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
Q1 YYYY

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
Extract spatio-temporal geographic relation quadruples from the text.

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
从以下文本中抽取地理关系四元组。

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
class GeoKGAttributeExtractorPrompt(PromptABC):
    """
    从地理文本中抽取属性四元组：

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
You are an expert in extracting spatio-temporal attribute quadruples from geographic text.

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

<subj> Mount Etna <attribute> elevation <value> 3329m <time> NA

Entity class:

["Volcano"]


CRITICAL CONSTRAINT:

If the correct leaf class cannot be determined,
DO NOT output the tuple.


=== TIME STANDARDIZATION ===

YYYY-MM-DD

Month YYYY

YYYY

Q1 YYYY

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
你是一名地理时空知识图谱属性抽取专家。

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

<subj> 埃特纳火山 <attribute> 海拔 <value> 3329m <time> NA

实体类别：

["Volcano"]


强约束：

如果无法确定实体对应的最小类别，
则不要输出该四元组。


=== 时间标准化 ===

YYYY-MM-DD
Month YYYY
YYYY
Q1 YYYY
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
Extract spatio-temporal attribute quadruples.

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
从以下文本中抽取属性四元组。

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


class GeoKGRelationInferencePrompt(PromptABC):
    """
    根据已有地理知识图谱四元组推断两个实体之间的关系：

    <subj> Entity <obj> Entity <rel> Relation <time> TimeValue
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
You are an expert in geographic knowledge graph reasoning.

Your task is to infer the relationship between two entities based on existing spatio-temporal quadruples.

Each quadruple follows the format:

<subj> Entity <obj> Entity <rel> Relation <time> TimeValue


=== INPUT ===

You will receive:

1. Two target entities
2. A set of existing quadruples related to them
3. A geographic ontology


=== ENTITY TYPES ===

Entities must belong to the following ontology leaf types:

{entity_str}


=== RELATION TYPES ===

Relations must be one of the following:

{relation_str}


=== REASONING RULES ===

Infer the most logical relation between the two entities using:

1. Transitive spatial reasoning  
Example:

A located_in B  
B located_in C  

Infer:

A located_in C


2. Geographic containment reasoning  

City located_in Province  
Province located_in Country  


3. Hydrological reasoning  

River flows_into River  
River flows_into Ocean


4. Proximity reasoning  

Village near Mountain


Only infer relations that are supported by the input tuples.


=== ENTITY CLASS RULES ===

You MUST output entity classes.

Rules:

- Classes must be ontology leaf types
- Order must match entity order


Example:

Tuple:

<subj> Wuhan <obj> China <rel> located_in <time> NA

Entity classes:

["City","Country"]


=== TIME RULES ===

If the inferred relation has no clear time:

Use NA


=== OUTPUT FORMAT ===

Return JSON ONLY:

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
你是一名地理知识图谱推理专家。

任务是根据已有四元组推断两个实体之间的关系。

四元组格式：

<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值


=== 输入 ===

你将获得：

1. 两个目标实体
2. 与它们相关的一组四元组
3. 一个地理本体库


=== 实体类型 ===

实体必须属于以下本体最小类别：

{entity_str}


=== 关系类型 ===

关系必须属于以下类别：

{relation_str}


=== 推理规则 ===

根据已有四元组推断关系：

1. 传递关系推理

示例：

A located_in B  
B located_in C  

推断：

A located_in C


2. 地理包含关系

City → Province → Country


3. 水文关系

River flows_into River  
River flows_into Ocean


4. 空间邻近关系

Village near Mountain


只能推断有证据支持的关系。


=== 实体类别规则 ===

必须输出实体类别：

规则：

- 必须是本体中的最小类别
- 顺序必须与实体顺序一致


示例：

<subj> 武汉 <obj> 中国 <rel> located_in <time> NA

实体类别：

["City","Country"]


=== 时间规则 ===

如果无法确定时间：

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


    def build_prompt(
        self,
        entity1: str,
        entity2: str,
        tuples: list
    ):

        tuple_text = "\n".join(tuples)

        if self.lang == "en":

            return textwrap.dedent(f"""
Infer the relationship between the following two entities.

Entity 1:
{entity1}

Entity 2:
{entity2}

Existing quadruples related to them:

{tuple_text}

Use the quadruples to infer the most logical relation.

Output JSON ONLY.
""")

        else:

            return textwrap.dedent(f"""
根据已有四元组推断以下两个实体之间的关系。

实体1：
{entity1}

实体2：
{entity2}

已有四元组：

{tuple_text}

请根据这些四元组推断两个实体之间最合理的关系。

仅输出 JSON。
""")


@PROMPT_REGISTRY.register()
class GeoKGEventExtractorPrompt(PromptABC):
    """
    从文本中抽取事件多元组（event tuples）：

    <event> EventDescription <location> Location <time> TimeValue <...> Optional fields
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self):
        if self.lang == "en":
            self.system_text = textwrap.dedent("""
You are an expert in extracting spatio-temporal events from geographic text.

Your task is to extract event tuples from input text.

Each tuple MUST contain:

<event> Event description
<location> Location
<time> Time of the event

Optionally, tuples may include other fields like <participants>, <cause>, <effect>, etc.

=== TIME STANDARDIZATION ===
1. Specific date: YYYY-MM-DD, e.g., 2025-03-03
2. Month: Full month name + year, e.g., March 2025
3. Year: YYYY, e.g., 2025
4. Quarter: QX YYYY, e.g., Q1 2025
5. Time interval: YYYY-MM-DD|YYYY-MM-DD, e.g., 2025-01-01|2025-01-03
6. If no time is explicitly mentioned in the text, set <time> to 'NA'

Rules:

1. Only extract information that appears in the text.
2. Do NOT invent events or locations.
3. Each tuple must contain <event>, <location>, and <time>.
4. Output JSON ONLY.
5. Each tuple should be self-contained and represent one event.

=== OUTPUT FORMAT ===

{
  "tuple": [
    "<event> ... <location> ... <time> ... <optional fields>"
  ]
}

Do NOT output explanations.
""")
        else:
            self.system_text = textwrap.dedent("""
你是一名地理文本事件抽取专家。

任务是从文本中抽取事件多元组。

每个元组必须包含：

<event> 事件描述
<location> 地点
<time> 时间

可选字段可以包括 <participants>、<cause>、<effect> 等。

=== 时间标准化 ===
1. 具体日期：YYYY-MM-DD，例如 2025-03-03
2. 月份：完整月份名称 + 年份，例如 March 2025
3. 年份：YYYY，例如 2025
4. 季度：QX YYYY，例如 Q1 2025
5. 时间区间：YYYY-MM-DD|YYYY-MM-DD，例如 2025-01-01|2025-01-03
6. 如果文本中没有明确时间，<time> 填入 'NA'

规则：

1. 仅抽取文本中出现的信息
2. 不得虚构事件或地点
3. 每个元组必须包含 <event>、<location> 和 <time>
4. 输出 JSON
5. 每个元组应独立，代表一个事件

=== 输出格式 ===

{
  "tuple": [
    "<event> ... <location> ... <time> ... <optional fields>"
  ]
}

不要输出解释。
""")
        return self.system_text

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""
Extract spatio-temporal event tuples from the following text according to the rules above.

Text:
{text}

Each tuple must contain <event>, <location>, and <time>.
Optional fields are allowed after these.

Output JSON ONLY:

{{
  "tuple": []
}}
""")
        else:
            return textwrap.dedent(f"""
从以下文本中抽取事件多元组。

文本：
{text}

每个元组必须包含 <event>、<location> 和 <time>。
可选字段可以跟在这些之后。

仅输出 JSON：

{{
  "tuple": []
}}
""")


# -*- coding: utf-8 -*-
@PROMPT_REGISTRY.register()
class GeoKGEventConsistencyPrompt(PromptABC):
    """
    Evaluate the consistency of event tuples.

    Each event tuple is formatted as:
        "<event> ... <location> ... <time> ... <cause> ... <effect> ..."

    The model should score each event tuple independently based on how consistent
    and logically coherent the information is.

    Score range:
        0 = very inconsistent / contradictory
        1 = fully consistent and logically coherent
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名事件信息评估专家。
                你的任务是评估每条事件多元组的一致性和逻辑合理性。

                ### 判断标准
                - 事件描述是否自洽
                - 时间、地点、原因、结果是否逻辑一致
                - 信息之间是否存在矛盾或不合理之处
                - 不需要评估事件重要性，只关注内部一致性

                ### 输出格式
                仅返回 JSON：
                {
                    "consistency_scores": [float, float, ...]
                }

                每个事件多元组对应一个分数，范围 0-1：
                1 = 完全一致
                0 = 完全矛盾或不一致
                不要输出任何解释。
            """)
        else:
            return textwrap.dedent("""\
                You are an expert in event information evaluation.
                Your task is to assess the **consistency** and logical coherence of each event tuple.

                ### Evaluation Criteria
                - Is the event description internally consistent?
                - Are the time, location, cause, and effect logically coherent?
                - Are there any contradictions or unreasonable information?
                - Do NOT evaluate importance, only internal consistency.

                ### Output Format
                Return ONLY a JSON object:

                {
                    "consistency_scores": [float, float, ...]
                }

                Each score must correspond to one event tuple in order.
                Score range: 0-1
                1 = fully consistent
                0 = highly inconsistent
                Do not output explanations.
            """)

    def build_prompt(self, event_tuples: list) -> str:
        """
        Format event tuples for LLM evaluation.

        Args:
            event_tuples (list): list of event tuple strings
        """
        event_block = ""
        for idx, ev in enumerate(event_tuples):
            event_block += f"ID {idx}: {ev}\n"

        if self.lang == "zh":
            return f"""请评估以下事件多元组的信息一致性。

            --- Event Tuples ---
            {event_block}

            请返回每条事件多元组的一致性分数（0-1），严格按照 JSON 输出。"""
        else:
            return f"""Evaluate the internal consistency of the following event tuples.

            --- Event Tuples ---
            {event_block}

            Return ONLY a JSON object containing consistency scores for each event tuple (0-1)."""


# -*- coding: utf-8 -*-
@PROMPT_REGISTRY.register()
class GeoKGQuadrupleRationalePrompt(PromptABC):
    """
    Evaluate the plausibility of quadruples (event tuples with 4 components).

    Each quadruple is formatted as:
        "<event> ... <location> ... <time> ... <cause/effect> ..."

    The model should score each quadruple independently based on how plausible
    and logically reasonable the information is.

    Score range:
        0 = very implausible / contradictory
        1 = fully plausible and reasonable
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名事件四元组评估专家。
                你的任务是评估每条事件四元组的合理性和逻辑性。

                ### 判断标准
                - 事件描述是否合理
                - 时间、地点、原因或结果是否符合常理
                - 信息之间是否存在矛盾或不合理之处
                - 不需要评估事件重要性，只关注逻辑合理性

                ### 输出格式
                仅返回 JSON：
                {
                    "rationale_scores": [float, float, ...]
                }

                每条四元组对应一个分数，范围 0-1：
                1 = 完全合理
                0 = 完全不合理或矛盾
                不要输出任何解释。
            """)
        else:
            return textwrap.dedent("""\
                You are an expert in evaluating event quadruples.
                Your task is to assess the **plausibility** and logical reasonableness of each quadruple.

                ### Evaluation Criteria
                - Is the event description reasonable?
                - Do time, location, cause, and effect follow common sense and logic?
                - Are there any contradictions or implausible information?
                - Do NOT evaluate importance, only logical plausibility.

                ### Output Format
                Return ONLY a JSON object:

                {
                    "rationale_scores": [float, float, ...]
                }

                Each score must correspond to one quadruple in order.
                Score range: 0-1
                1 = fully plausible
                0 = highly implausible or contradictory
                Do not output explanations.
            """)

    def build_prompt(self, quadruples: list) -> str:
        """
        Format quadruples for LLM evaluation.

        Args:
            quadruples (list): list of quadruple strings
        """
        quad_block = ""
        for idx, quad in enumerate(quadruples):
            quad_block += f"ID {idx}: {quad}\n"

        if self.lang == "zh":
            return f"""请评估以下事件四元组的合理性和逻辑性。

            --- Quadruples ---
            {quad_block}

            请返回每条四元组的合理性分数（0-1），严格按照 JSON 输出。"""
        else:
            return f"""Evaluate the plausibility and logical reasonableness of the following event quadruples.

            --- Quadruples ---
            {quad_block}

            Return ONLY a JSON object containing plausibility scores for each quadruple (0-1)."""