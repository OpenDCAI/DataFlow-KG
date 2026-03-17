import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class LegalKGRelationExtractorPrompt(PromptABC):

    def __init__(self, lang: str = "zh"):
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

        self.system_text = textwrap.dedent(f"""
你是一名法律知识图谱关系抽取专家。

任务包括两部分：
1）抽取关系三元组
2）生成案件简要摘要


====================
一、关系三元组抽取
====================

格式：

<subj> 实体 <obj> 实体 <rel> 关系


=== 实体规则 ===
{entity_str}

- 仅使用文本中出现的实体
- 不得虚构
- 不得使用代词
- 必须为最小类别


=== 关系规则 ===
{relation_str}

- 不得虚构关系
- 必须使用本体关系


=== 实体类别 ===

每个三元组必须输出：

["头实体类别","尾实体类别"]

必须为最小类别


====================
二、案件摘要（case_summary）
====================

要求：

- 用1~3句话总结案件
- 包含：当事人、行为、结果
- 简洁、客观
- 不要编造信息


====================
输出格式（严格）
====================

仅输出 JSON：

{{
  "triple":[
    "<subj> 实体 <obj> 实体 <rel> 关系"
  ],
  "entity_class":[
    ["头实体类别","尾实体类别"]
  ],
  "case_summary":"案件摘要"
}}

禁止输出解释文本。
""")

        return self.system_text

    def build_prompt(self, text: str):

        return textwrap.dedent(f"""
从以下法律文本中抽取关系三元组并生成案件摘要：

{text}

仅输出 JSON：
""")


@PROMPT_REGISTRY.register()
class LegalKGAttributeExtractorPrompt(PromptABC):

    def __init__(self, lang: str = "zh"):
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

        self.system_text = textwrap.dedent(f"""
你是一名法律知识图谱属性抽取专家。

任务包括两部分：
1）抽取属性三元组
2）生成案件简要摘要


====================
一、属性三元组抽取
====================

格式：

<subj> 实体 <attribute> 属性 <value> 属性值


=== 实体规则 ===
{entity_str}

=== 属性规则 ===
{attribute_str}


=== 实体类别 ===

每个三元组必须输出：

["实体类别"]

必须为最小类别


====================
二、案件摘要（case_summary）
====================

要求：

- 用1~3句话总结案件
- 包含：当事人、关键行为、裁判结果
- 简洁客观


====================
输出格式（严格）
====================

仅输出 JSON：

{{
  "triple":[
    "<subj> 实体 <attribute> 属性 <value> 属性值"
  ],
  "entity_class":[
    ["实体类别"]
  ],
  "case_summary":"案件摘要"
}}

禁止输出解释文本。
""")

        return self.system_text

    def build_prompt(self, text: str):

        return textwrap.dedent(f"""
从以下法律文本中抽取属性三元组并生成案件摘要：

{text}

仅输出 JSON：
""")