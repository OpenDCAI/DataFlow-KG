import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class SchoKGRelationExtractorPrompt(PromptABC):
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
                You are an expert in extracting scholarly knowledge graph relations from text.
                You are given a predefined ontology specifying valid entities and relations.

                === RULES ===
                1. ENTITY:
                   - Must be one of the following bottom-level types ONLY:
                     {entity_str}
                   - No pronouns or invented entities
                   - Do NOT use high-level categories like Person or Publication
                2. RELATION:
                   - Must be one of the following bottom-level relations ONLY:
                     {relation_str}
                   - Do NOT use high-level relation categories like AuthorshipRelation
                3. FACT:
                   - Each triple represents ONE factual relation
                   - Ignore information outside the scholarly domain

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
                Extract scholarly knowledge graph relations from the following text.
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
                    ["Author", "University"],
                    ["Paper", "Conference"]
                  ]
                }}
                Example:
                {{
                  "triple": [
                    "<subj> Geoffrey Hinton <obj> University of Toronto <rel> affiliated_with",
                    "<subj> Attention Is All You Need <obj> NeurIPS <rel> published_in"
                  ],
                  "entity_class": [
                    ["Author", "University"],
                    ["Paper", "Conference"]
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
class SchoKGQueryReasoningPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self):
        if self.lang == "zh":
            self.system_text = textwrap.dedent("""\
                你是一个学者知识图谱问答助手。
                你会得到用户问题和若干候选路径。

                你的任务：
                1. 从候选路径中选出最能支持回答的问题路径
                2. 基于这些路径生成简洁、准确的回答

                规则：
                1. 只能依据给定候选路径回答，不能编造新事实
                2. 如果候选路径不足以支持回答，就返回空路径并明确说明证据不足
                3. 优先选择与问题语义最直接相关的路径
                4. 只输出 JSON，不要输出额外解释

                输出格式：
                {
                  "reasoning_path": [
                    "<subj> entity1 <obj> entity2 <rel> relation || <subj> entity2 <obj> entity3 <rel> relation"
                  ],
                  "reasoning_answer": "..."
                }
            """)
        else:
            self.system_text = textwrap.dedent("""\
                You are a scholarly knowledge graph question answering assistant.
                You will be given a user query and several candidate paths.

                Your tasks:
                1. Select the candidate paths that best support answering the query
                2. Generate a concise and accurate answer based on the selected paths

                Rules:
                1. Answer only based on the provided candidate paths and do not invent new facts
                2. If the candidate paths are insufficient, return an empty path list and clearly say the evidence is insufficient
                3. Prefer paths that are most directly relevant to the query
                4. Output JSON only, with no extra explanation

                Output format:
                {
                  "reasoning_path": [
                    "<subj> entity1 <obj> entity2 <rel> relation || <subj> entity2 <obj> entity3 <rel> relation"
                  ],
                  "reasoning_answer": "..."
                }
            """)

        return self.system_text

    def build_prompt(self, query: str, candidate_paths: list):
        candidate_path_text = json.dumps(candidate_paths, ensure_ascii=False, indent=2)

        if self.lang == "zh":
            return textwrap.dedent(f"""\
                用户问题：
                {query}

                候选路径：
                {candidate_path_text}

                请从候选路径中选出最能回答该问题的路径，并生成回答。
                只输出 JSON：
                {{
                  "reasoning_path": [
                    "<subj> ... <obj> ... <rel> ... || <subj> ... <obj> ... <rel> ..."
                  ],
                  "reasoning_answer": "..."
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
              "reasoning_path": [
                "<subj> ... <obj> ... <rel> ... || <subj> ... <obj> ... <rel> ..."
              ],
              "reasoning_answer": "..."
            }}
        """)


@PROMPT_REGISTRY.register()
class SchoKGRecommendPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = None

    def build_system_prompt(self):
        if self.lang == "zh":
            self.system_text = textwrap.dedent("""\
                你是一个学者知识图谱节点推荐助手。
                你会得到用户问题、目标节点类型和候选节点及其支持路径。

                你的任务：
                1. 选择最适合推荐的节点
                2. 基于支持路径给出简洁、准确的推荐理由

                规则：
                1. 只能依据给定候选节点和支持路径推荐，不能编造新事实
                2. 优先选择与问题最直接相关、支持路径最充分的节点
                3. 如果证据不足，就返回空列表并明确说明证据不足
                4. 只输出 JSON，不要输出额外解释

                输出格式：
                {
                  "recommended_node": ["node1", "node2"],
                  "recommendation_reason": "..."
                }
            """)
        else:
            self.system_text = textwrap.dedent("""\
                You are a scholarly knowledge graph node recommendation assistant.
                You will be given a user query, a target node type, and candidate nodes with supporting paths.

                Your tasks:
                1. Select the best nodes to recommend
                2. Generate a concise and accurate recommendation reason based on the supporting paths

                Rules:
                1. Recommend only based on the provided candidate nodes and supporting paths; do not invent new facts
                2. Prefer nodes that are most directly relevant to the query and have stronger path support
                3. If the evidence is insufficient, return empty lists and clearly say the evidence is insufficient
                4. Output JSON only, with no extra explanation

                Output format:
                {
                  "recommended_node": ["node1", "node2"],
                  "recommendation_reason": "..."
                }
            """)

        return self.system_text

    def build_prompt(self, query: str, target_type: str, candidate_nodes: list):
        candidate_node_text = json.dumps(candidate_nodes, ensure_ascii=False, indent=2)

        if self.lang == "zh":
            return textwrap.dedent(f"""\
                用户问题：
                {query}

                目标节点类型：
                {target_type}

                候选节点：
                {candidate_node_text}

                请从候选节点中选出最值得推荐的节点，并生成推荐理由。
                只输出 JSON：
                {{
                  "recommended_node": ["node1", "node2"],
                  "recommendation_reason": "..."
                }}
            """)

        return textwrap.dedent(f"""\
            User query:
            {query}

            Target node type:
            {target_type}

            Candidate nodes:
            {candidate_node_text}

            Select the best nodes to recommend and generate the recommendation reason.
            Output JSON only:
            {{
              "recommended_node": ["node1", "node2"],
              "recommendation_reason": "..."
            }}
        """)
