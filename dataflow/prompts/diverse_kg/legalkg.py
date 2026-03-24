import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
from typing import Any, Dict, List, Union


@PROMPT_REGISTRY.register()
class LegalKGRelationExtractorPrompt(PromptABC):
    """
    从法律文本中抽取关系三元组并生成案件摘要

    输出格式:
    {
      "triple":[
        "<subj> 主体 <obj> 客体 <rel> 中文关系"
      ],
      "entity_class":[
        ["头实体类别","尾实体类别"]
      ],
      "case_summary":"案件摘要"
    }
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        # 获取所有实体和关系叶节点
        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        relation_str = ", ".join(relation_list)

        if self.lang == "zh":
            self.system_text = textwrap.dedent(f"""
你是一名法律知识图谱关系抽取专家。

任务：
1）抽取关系三元组
2）生成案件摘要

====================
一、关系三元组抽取
====================

三元组格式（严格遵守）：

<subj> 主体 <obj> 客体 <rel> 关系

示例：
<subj> 刘某 <obj> 盗窃罪 <rel> 构成

⚠️ 严格约束：
- 关系必须使用中文本体关系
- 不输出“实体”“关系”等多余词
- 标签 <subj>/<obj>/<rel> 不可修改

=== 实体规则 ===
{entity_str}
- 仅使用文本中出现的实体
- 不得使用代词
- 不得虚构
- 必须为最小类别

=== 关系规则 ===
{relation_str}
- 必须严格匹配本体关系
- 不得改写
- 不在列表中则不输出

=== 关系方向约束 ===
- 被告人 → 构成 → 罪名
- 检察院 → 起诉 → 被告人
- 法院 → 判决 → 被告人
方向错误 → 不输出

=== 属性过滤 ===
以下不得作为关系：
- 金额
- 刑期
- 罚金
- 时间

=== 实体类别 ===
每个三元组必须输出：
["头实体类别","尾实体类别"]
- 必须为最小类别
- 顺序必须一致

====================
二、案件摘要
====================
要求：
- 1~3句话
- 包含当事人、行为、判决结果
- 客观简洁
- 不得编造

====================
输出格式（严格）
====================
仅输出 JSON：
{{
  "triple":[
    "<subj> 主体 <obj> 客体 <rel> 中文关系"
  ],
  "entity_class":[
    ["头实体类别","尾实体类别"]
  ],
  "case_summary":"案件摘要"
}}
禁止输出解释文本。
""")
        else:
            # 英文版本
            self.system_text = textwrap.dedent(f"""
You are an expert in legal knowledge graph extraction.

Tasks:
1) Extract relation triples
2) Generate a case summary

====================
TRIPLE EXTRACTION
====================
Format:
<subj> Subject <obj> Object <rel> Relation

Example:
<subj> Liu Mou <obj> Theft <rel> constitutes

⚠️ HARD CONSTRAINTS:
- Relations must be from ENGLISH ontology
- Do NOT output Chinese relations
- Do NOT output extra words
- Do NOT modify tags <subj>/<obj>/<rel>

=== ENTITY RULES ===
{entity_str}
- Only use entities from the text
- No pronouns
- No hallucination
- Must be leaf types

=== RELATION RULES ===
{relation_str}
- Must exactly match ontology
- No paraphrase
- Out-of-list → skip

=== DIRECTION CONSTRAINT ===
- Defendant → constitutes → Crime
- Procuratorate → prosecutes → Defendant
- Court → judges → Defendant
Wrong direction → discard

=== ATTRIBUTE FILTER ===
Do NOT treat as relations:
- amount
- sentence
- fine
- time

=== ENTITY CLASS ===
Each triple must output:
["HeadEntityClass","TailEntityClass"]
- Must be leaf types
- Order must match

====================
CASE SUMMARY
====================
- 1–3 sentences
- Include parties, behavior, judgment
- Objective, no hallucination

====================
OUTPUT FORMAT
====================
Return JSON ONLY:
{{
  "triple":[
    "<subj> Subject <obj> Object <rel> Relation"
  ],
  "entity_class":[
    ["HeadEntityClass","TailEntityClass"]
  ],
  "case_summary":"summary"
}}
Do NOT output explanations.
""")
        return self.system_text

    def build_prompt(self, text: str):
        if self.lang == "zh":
            return textwrap.dedent(f"""
从以下法律文本中抽取关系三元组并生成案件摘要：

{text}

仅输出 JSON：
""")
        else:
            return textwrap.dedent(f"""
Extract relation triples and generate a case summary:

{text}

Return JSON ONLY:
""")



@PROMPT_REGISTRY.register()
class LegalKGAttributeExtractorPrompt(PromptABC):
    """
    从法律文本中抽取属性三元组并生成案件摘要

    输出格式:
    {
      "triple":[
        "<entity> 主体 <attribute> 属性 <value> 属性值"
      ],
      "entity_class":[
        ["实体类别"]
      ],
      "case_summary":"案件摘要"
    }
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang.lower()
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        # 获取本体实体与属性叶子节点
        entity_leaf_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_leaf_list.extend(group)

        attribute_list = []
        for group in ontology.get("attribute_type", {}).values():
            attribute_list.extend(group)

        entity_str = ", ".join(entity_leaf_list)
        attribute_str = ", ".join(attribute_list)

        if self.lang == "zh":
            self.system_text = textwrap.dedent(f"""
你是一名法律知识图谱属性抽取专家。

任务：
1）抽取属性三元组（严格按指定格式）
2）生成案件摘要（中文）

====================
一、属性三元组抽取
====================

格式（必须严格遵守，标签不可替换）：
<entity> 实体 <attribute> 属性 <value> 属性值

说明：
- <entity>/<attribute>/<value> 为固定标签，占位符不可替换
- 实体、属性、值写在标签后面，用空格分隔
- 不输出额外字符或标签

示例：
<entity> 检察院 <attribute> 案号 <value> 沪黄检刑诉〔2025〕13号
<entity> 刘某 <attribute> 刑期 <value> 有期徒刑三年

=== 实体规则 ===
{entity_str}
- 仅使用文本中出现的实体
- 不得使用代词
- 不得虚构
- 必须为最小类别
- 必须用中文

=== 属性规则 ===
{attribute_str}
- 必须严格匹配本体属性
- 不得改写
- 不在列表中则不输出
- 必须用中文

=== 属性归属约束 ===
- 属性必须归属于合理实体：
  - 被告人 → 刑期 / 罚金
  - 案件 → 案号 / 审理程序
  - 行为 → 金额 / 手段
归属错误 → 不输出

=== 与关系区分 ===
以下必须作为属性（不能当关系）：
- 金额、刑期、罚金、时间、比例

=== 实体类别 ===
每个三元组对应实体类别，格式：
["实体类别"]
顺序与 triple 列表一致

====================
二、案件摘要
====================
- 1~3句话
- 必须中文，不能输出英文摘要
- 包含当事人、关键行为、裁判结果
- 客观简洁
- 不得虚构信息

====================
输出格式
====================
仅输出 JSON：
{{
  "triple":[
    "<entity> 实体 <attribute> 属性 <value> 属性值"
  ],
  "entity_class":[
    ["实体类别"]
  ],
  "case_summary":"案件摘要"
}}
禁止输出解释文本。
""")
        else:
            self.system_text = textwrap.dedent(f"""
You are a legal knowledge graph attribute extraction expert.

Tasks:
1) Extract attribute triples (strictly follow the format)
2) Generate a case summary (English)

====================
ATTRIBUTE EXTRACTION
====================

Format (tags MUST NOT be replaced):
<entity> Entity <attribute> Attribute <value> Value

Explanation:
- <entity>/<attribute>/<value> are fixed placeholders, do not replace them
- Entity, attribute, and value are written after the tags, separated by spaces
- No extra characters or labels

Example:
<entity> Procuratorate <attribute> case_number <value> HUANGHUANGJIANXINGSU〔2025〕No.13
<entity> Liu <attribute> sentence <value> 3 years in prison

=== ENTITY RULES ===
{entity_str}
- Only use entities present in text
- No pronouns
- No hallucination
- Must be leaf types

=== ATTRIBUTE RULES ===
{attribute_str}
- Must exactly match ontology
- Do not paraphrase
- Skip if not in ontology

=== ATTRIBUTE OWNERSHIP ===
- Must belong to correct entities:
  - Defendant → sentence / fine
  - Case → case_id / procedure
  - Action → amount / means
Wrong assignment → discard

=== RELATION FILTER ===
The following MUST be attributes (NOT relations):
- amount, sentence, fine, time, ratio

=== ENTITY CLASS ===
Each triple must output leaf-level entity class in ["EntityClass"], order matches triple list

====================
CASE SUMMARY
====================
- 1–3 sentences
- Must be English
- Include parties, key actions, judgment
- Concise, objective, no hallucination

====================
Output format
====================
Return JSON ONLY:
{{
  "triple":[
    "<entity> Entity <attribute> Attribute <value> Value"
  ],
  "entity_class":[
    ["EntityClass"]
  ],
  "case_summary":"English case summary"
}}
Do NOT output explanations.
""")

        return self.system_text

    def build_prompt(self, text: str):
        if self.lang == "zh":
            return textwrap.dedent(f"""
请从以下法律文本中抽取属性三元组并生成中文案件摘要：

{text}

仅输出 JSON：
""")
        else:
            return textwrap.dedent(f"""
Extract attribute triples and generate an English case summary from the following legal text:

{text}

Return JSON ONLY:
""")


@PROMPT_REGISTRY.register()
class CaseSummarySimilarityPrompt(PromptABC):
    """
    评估案件摘要与案件类型描述的语义相似度。
    输出 JSON:
    {
        "similarity_score": <float>  // 范围 0-1
    }
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名法律文本语义分析专家。
                任务：评估案件摘要与案件类型描述的匹配程度（0-1）。

                ### 要求：
                - 输入 case_summary 是案件摘要文本
                - 输入 case_type 是用户输入的案件类型描述（如“盗窃案件”）
                - 输出一个浮点数，表示摘要与类型描述的匹配程度
                  1表示完全匹配，0表示完全不匹配
                - 仅输出 JSON：
                {
                    "similarity_score": <float>
                }
                禁止输出其他解释或文本。
            """)
        else:
            return textwrap.dedent("""\
                You are an expert in legal text semantic analysis.
                Task: Evaluate the similarity between a case summary and a case type description (0-1).

                ### Requirements:
                - Input 'case_summary' is the textual summary of the case
                - Input 'case_type' is the type description (e.g., "theft case")
                - Output a float representing similarity: 1 = fully matching, 0 = completely unrelated
                - Return JSON only:
                {
                    "similarity_score": <float>
                }
                Do not output any explanation or extra text.
            """)

    def build_prompt(self, case_summary: str, case_type: str) -> str:
        if self.lang == "zh":
            return textwrap.dedent(f"""
请评估以下案件摘要与案件类型描述的匹配程度（0-1）。

案件摘要：
{case_summary}

案件类型描述：
{case_type}

请仅返回 JSON 格式：
{{
  "similarity_score": <float>
}}
""")
        else:
            return textwrap.dedent(f"""
Evaluate the similarity (0-1) between the following case summary and case type description.

Case Summary:
{case_summary}

Case Type Description:
{case_type}

Return JSON only:
{{
  "similarity_score": <float>
}}
""")


@PROMPT_REGISTRY.register()
class LegalKGJudgementPredictionPrompt(PromptABC):
    """
    根据三元组预测判决，并返回支撑三元组
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名法律判决推理专家。

                你的任务：
                基于给定的知识图谱三元组（triple）和案件描述（case_description），预测案件判决，并给出支撑依据。

                ====================
                任务要求
                ====================

                1）判决预测（judgement）
                - 给出合理的法律判决结果
                - 必须简洁明确（如：有期徒刑三年，并处罚金）

                2）理由（reason）
                - 必须从输入 triple 中选择
                - 不允许编造新的三元组
                - 必须是与判决直接相关的关键事实

                ====================
                输出格式（严格）
                ====================
                仅输出 JSON：

                {
                  "judgement": "判决结果",
                  "reason": [
                    "<entity> ... <attribute> ... <value> ..."
                  ]
                }

                禁止输出解释文本。
            """)
        else:
            return textwrap.dedent("""\
                You are a legal reasoning expert.

                Task:
                Given a set of triples and a case description, predict the judgement and provide supporting triples.

                ====================
                Requirements
                ====================

                1) Judgement
                - Provide a concise legal decision

                2) Reason
                - MUST be selected from input triples
                - NO hallucination
                - Only include triples directly supporting the judgement

                ====================
                Output Format
                ====================
                Return JSON ONLY:

                {
                  "judgement": "predicted judgement",
                  "reason": [
                    "<entity> ... <attribute> ... <value> ..."
                  ]
                }
            """)

    def build_prompt(self, triples: List[str], case_desc: str) -> str:
        triple_block = "\n".join(triples)

        if self.lang == "zh":
            return textwrap.dedent(f"""
请根据以下三元组和案件描述预测判决结果：

【案件描述】
{case_desc}

【知识图谱三元组】
{triple_block}

请输出判决结果和支撑三元组（必须从上面选择）。
仅输出JSON：
""")
        else:
            return textwrap.dedent(f"""
Predict the judgement based on triples and case description.

Case Description:
{case_desc}

Triples:
{triple_block}

Return judgement and supporting triples in JSON only.
""")