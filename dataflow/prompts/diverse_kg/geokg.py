import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json

@PROMPT_REGISTRY.register()
class GeoKGRelationExtractorPrompt(PromptABC):
    """
    从地理文本中抽取关系三元组，实体和关系必须来源于预定义本体底层类别
    输出格式: <subj> 实体 <obj> 实体 <rel> 关系
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        """
        构建系统提示，明确要求只使用最底层实体和关系
        """
        entity_list = []
        for group in ontology.get("entity_type", {}).values():
            entity_list.extend(group)

        relation_list = []
        for group in ontology.get("relation_type", {}).values():
            relation_list.extend(group)

        entity_str = ", ".join(entity_list)
        relation_str = ", ".join(relation_list)

        if self.lang == "en":
            self.system_text = textwrap.dedent(f"""\
                You are an expert in extracting geographical knowledge graph relations from text.
                You are given a predefined ontology specifying valid entities and relations.

                === RULES ===
                1. ENTITY:
                   - Must be one of the following bottom-level types ONLY:
                     {entity_str}
                   - No pronouns or invented entities
                   - Do NOT use high-level categories like NaturalFeature or AdministrativeRegion
                2. RELATION:
                   - Must be one of the following bottom-level relations ONLY:
                     {relation_str}
                   - Do NOT use high-level relation categories like SpatialRelation
                3. FACT:
                   - Each triple represents ONE factual relation
                   - Ignore information outside geographical domain

                === OUTPUT FORMAT ===
                - Only output JSON
                - Key: "tuple"
                - Each item is a string:
                  "<subj> Entity <obj> Entity <rel> Relation"
                - Do NOT add explanations or extra text
            """)
        else:
            self.system_text = textwrap.dedent(f"""\
                你是一名地理知识图谱关系抽取专家。
                已知预定义本体文件，包含有效实体和关系的底层类别。

                === 规则 ===
                1. 实体：
                   - 必须是以下底层类型之一：
                     {entity_str}
                   - 禁止使用代词或虚构实体
                   - 不得使用高层类别名称，如 NaturalFeature 或 AdministrativeRegion
                2. 关系：
                   - 必须是以下底层关系之一：
                     {relation_str}
                   - 不得使用高层类别名称，如 SpatialRelation
                3. 事实：
                   - 每条三元组表达一个事实
                   - 忽略非地理领域信息

                === 输出格式 ===
                - 仅输出 JSON
                - 键名为 "tuple"
                - 每条为字符串：
                  "<subj> 实体 <obj> 实体 <rel> 关系"
                - 不输出解释或其他文本
            """)
        return self.system_text

    def build_prompt(self, text: str):
        """
        构建用户提示，强调必须使用本体底层实体和关系
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract geographical knowledge graph relations from the following text.
                Use ONLY entities and relations specified in the system prompt ontology.
                Do NOT invent any entity or relation.

                Text:
                {text}

                Output ONLY JSON:
                {{
                  "tuple": [
                    "<subj> Entity <obj> Entity <rel> Relation",
                    "<subj> Entity <obj> Entity <rel> Relation"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                从以下文本中抽取地理知识图谱关系。
                仅使用系统提示中本体定义的底层实体和关系。
                不得虚构任何实体或关系。

                文本：
                {text}

                仅输出 JSON：
                {{
                  "tuple": [
                    "<subj> 实体 <obj> 实体 <rel> 关系",
                    "<subj> 实体 <obj> 实体 <rel> 关系"
                  ]
                }}
            """)