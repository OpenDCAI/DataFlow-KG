import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class MedKGRelationExtractorPrompt(PromptABC):
    """
    从医学文本中抽取关系三元组，实体和关系必须来自预定义本体的底层类别。
    输出格式: <subj> 实体 <obj> 实体 <rel> 关系
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self, ontology: dict):
        """
        构建系统提示，明确要求只使用本体底层实体和关系。
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
                You are an expert in extracting medical knowledge graph relations from text.
                You are given a predefined ontology specifying valid entities and relations.

                === RULES ===
                1. ENTITY:
                   - Must be one of the following bottom-level types ONLY:
                     {entity_str}
                   - No pronouns or invented entities
                   - Do NOT use high-level categories like Substance_and_Drug
                2. RELATION:
                   - Must be one of the following bottom-level relations ONLY:
                     {relation_str}
                   - Do NOT use high-level relation categories like Anatomy-Gene Relation
                3. FACT:
                   - Each triple represents ONE factual relation
                   - Ignore information outside medical domain

                === OUTPUT FORMAT ===
                - Only output JSON
                - Key: "triple"
                - Key: "entity_class"
                - "triple": each item is a string:
                  "<subj> subject <obj> object <rel> relation"
                - "entity_class": each item is a list of bottom-level entity types
                - The i-th entity_class item must correspond to the i-th triple
                - Each entity_class item should contain the subject/object entity types used in that triple
                - You must keep the literal markers "<subj>", "<obj>", and "<rel>" exactly as written
                - Do NOT output words like "Entity", or "Relation" in the triple string
                - Output only the subject text, object text, and relation label
                - Do NOT add explanations or extra text
            """)
        else:
            self.system_text = textwrap.dedent(f"""\
                你是一名医学知识图谱关系抽取专家。
                已知预定义本体文件，包含有效实体和关系的底层类别。

                === 规则 ===
                1. 实体：
                   - 必须是以下底层类型之一：
                     {entity_str}
                   - 禁止使用代词或虚构实体
                   - 不得使用高层类别名称，如 Substance_and_Drug
                2. 关系：
                   - 必须是以下底层关系之一：
                     {relation_str}
                   - 不得使用高层类别名称，如 Anatomy-Gene Relation
                3. 事实：
                   - 每条三元组表达一个事实
                   - 忽略非医学领域信息

                === 输出格式 ===
                - 仅输出 JSON
                - 键名为 "triple"
                - 键名为 "entity_class"
                - "triple" 中每项为字符串：
                  "<subj> 主体 <obj> 客体 <rel> 关系"
                - "entity_class" 中每项为底层实体类型列表
                - 第 i 项 entity_class 必须对应第 i 项 triple
                - 每项 entity_class 应包含该 triple 中主体和客体的底层实体类型
                - 必须原样保留字面量 "<subj>"、"<obj>" 和 "<rel>"
                - 不要在 triple 字符串中输出 "Entity" 或 "Relation"
                - 只输出主体文本、客体文本和关系标签
                - 不要输出解释或其他文本
            """)
        return self.system_text

    def build_prompt(self, text: str):
        """
        构建用户提示，强调必须使用本体底层实体和关系。
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract medical knowledge graph relations from the following text.
                Use ONLY entities and relations specified in the system prompt ontology.
                Do NOT invent any entity or relation.

                Text:
                {text}

                Output ONLY JSON:
                {{
                  "triple": [
                    "<subj> subject <obj> object <rel> relation",
                    "<subj> subject <obj> object <rel> relation"
                  ],
                  "entity_class": [
                    ["Disease", "Anatomy"],
                    ["Gene", "Gene"]
                  ]
                }}
                Example:
                {{
                  "triple": [
                    "<subj> non-small cell lung cancer <obj> lung <rel> localizes",
                    "<subj> TP53 <obj> EGFR <rel> expresses"
                  ],
                  "entity_class": [
                    ["Disease", "Anatomy"],
                    ["Gene", "Gene"]
                  ]
                }}
                Keep the literal markers "<subj>", "<obj>", and "<rel>" exactly as written in every triple string.
                Do not output words like "entity" or "relation" in the triple string.
                The "entity_class" list must align with the "triple" list item by item.
            """)
        else:
            return textwrap.dedent(f"""\
                从以下文本中抽取医学知识图谱关系。
                仅使用系统提示中本体定义的底层实体和关系。
                不得虚构任何实体或关系。

                文本：
                {text}

                仅输出 JSON：
                {{
                  "triple": [
                    "<subj> 主体 <obj> 客体 <rel> 关系",
                    "<subj> 主体 <obj> 客体 <rel> 关系"
                  ],
                  "entity_class": [
                    ["Disease", "Anatomy"],
                    ["Gene", "Gene"]
                  ]
                }}
                示例：
                {{
                  "triple": [
                    "<subj> non-small cell lung cancer <obj> lung <rel> localizes",
                    "<subj> TP53 <obj> EGFR <rel> expresses"
                  ],
                  "entity_class": [
                    ["Disease", "Anatomy"],
                    ["Gene", "Gene"]
                  ]
                }}
                每条 triple 都必须原样保留 "<subj>"、"<obj>" 和 "<rel>"。
                不要在 triple 字符串中输出 "entity" 或 "relation" 这类标签。
                "entity_class" 列表必须与 "triple" 列表逐项对应。
            """)


@PROMPT_REGISTRY.register()
class MedKGDrugActionMechanismPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self):
        if self.lang == "zh":
            self.system_text = textwrap.dedent("""\
                你是一名医学知识图谱推理助手。
                你将获得用户查询和若干候选路径。

                你的任务：
                1. 从候选路径中选择最能回答查询的路径
                2. 基于选中的路径生成简洁、准确的自然语言回答

                规则：
                1. 只能依据输入中提供的候选路径回答，不能虚构新事实
                2. 如果候选路径不足以支持回答，就返回空列表并明确说明证据不足
                3. 优先选择与查询主题最相关、语义最直接的路径
                4. 只输出 JSON，不要输出额外解释

                输出格式：
                {
                  "mechanism_path": [
                    "<subj> entity1 <obj> entity2 <rel> relation || <subj> entity2 <obj> entity3 <rel> relation"
                  ],
                  "mechanism_answer": "..."
                }
            """)
        else:
            self.system_text = textwrap.dedent("""\
                You are a medical knowledge graph reasoning assistant.
                You will be given a user query and several candidate paths.

                Your tasks:
                1. Select the candidate paths that best answer the query
                2. Generate a concise and accurate natural language answer based on the selected paths

                Rules:
                1. Answer only based on the provided candidate paths and do not invent new facts
                2. If the candidate paths are insufficient, return an empty list and clearly say the evidence is insufficient
                3. Prefer paths that are most directly relevant to the query
                4. Output JSON only, with no extra explanation

                Output format:
                {
                  "mechanism_path": [
                    "<subj> entity1 <obj> entity2 <rel> relation || <subj> entity2 <obj> entity3 <rel> relation"
                  ],
                  "mechanism_answer": "..."
                }
            """)

        return self.system_text

    def build_prompt(self, query: str, candidate_paths: list):
        candidate_path_text = json.dumps(candidate_paths, ensure_ascii=False, indent=2)

        if self.lang == "zh":
            return textwrap.dedent(f"""\
                用户查询：
                {query}

                候选路径：
                {candidate_path_text}

                请从候选路径中选择最能回答该查询的路径，并生成回答。
                只输出 JSON：
                {{
                  "mechanism_path": [
                    "<subj> ... <obj> ... <rel> ... || <subj> ... <obj> ... <rel> ..."
                  ],
                  "mechanism_answer": "..."
                }}
            """)

        return textwrap.dedent(f"""\
            User query:
            {query}

            Candidate paths:
            {candidate_path_text}

            Select the candidate paths that best answer the query and generate the answer.
            Output JSON only:
            {{
              "mechanism_path": [
                "<subj> ... <obj> ... <rel> ... || <subj> ... <obj> ... <rel> ..."
              ],
              "mechanism_answer": "..."
            }}
        """)


@PROMPT_REGISTRY.register()
class MedKGDrugRepositioningPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self):
        if self.lang == "zh":
            self.system_text = textwrap.dedent("""\
                你是一名医学知识图谱药物重定位助手。
                你将获得用户查询和若干候选路径。

                你的任务：
                1. 从候选路径中选择最能支持药物重定位判断的路径
                2. 基于这些路径生成简洁、准确的自然语言回答

                规则：
                1. 只能依据输入中提供的候选路径回答，不能虚构新事实
                2. 优先选择能体现“药物可能作用于某个新疾病”的路径
                3. 如果候选路径不足以支持药物重定位判断，就返回空列表并明确说明证据不足
                4. 只输出 JSON，不要输出额外解释

                输出格式：
                {
                  "reposition_path": [
                    "<subj> entity1 <obj> entity2 <rel> relation || <subj> entity2 <obj> entity3 <rel> relation"
                  ],
                  "reposition_answer": "..."
                }
            """)
        else:
            self.system_text = textwrap.dedent("""\
                You are a medical knowledge graph assistant for drug repositioning.
                You will be given a user query and several candidate paths.

                Your tasks:
                1. Select the candidate paths that best support a drug repositioning hypothesis
                2. Generate a concise and accurate natural language answer based on those paths

                Rules:
                1. Answer only based on the provided candidate paths and do not invent new facts
                2. Prefer paths that suggest the drug may be useful for a new disease indication
                3. If the candidate paths are insufficient for repositioning, return an empty list and clearly say the evidence is insufficient
                4. Output JSON only, with no extra explanation

                Output format:
                {
                  "reposition_path": [
                    "<subj> entity1 <obj> entity2 <rel> relation || <subj> entity2 <obj> entity3 <rel> relation"
                  ],
                  "reposition_answer": "..."
                }
            """)

        return self.system_text

    def build_prompt(self, query: str, candidate_paths: list):
        candidate_path_text = json.dumps(candidate_paths, ensure_ascii=False, indent=2)

        if self.lang == "zh":
            return textwrap.dedent(f"""\
                用户查询：
                {query}

                候选路径：
                {candidate_path_text}

                请从候选路径中选择最能支持药物重定位的路径，并生成回答。
                只输出 JSON：
                {{
                  "reposition_path": [
                    "<subj> ... <obj> ... <rel> ... || <subj> ... <obj> ... <rel> ..."
                  ],
                  "reposition_answer": "..."
                }}
            """)

        return textwrap.dedent(f"""\
            User query:
            {query}

            Candidate paths:
            {candidate_path_text}

            Select the candidate paths that best support drug repositioning and generate the answer.
            Output JSON only:
            {{
              "reposition_path": [
                "<subj> ... <obj> ... <rel> ... || <subj> ... <obj> ... <rel> ..."
              ],
              "reposition_answer": "..."
            }}
        """)
