import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class KGQueryExtractionPrompt(PromptABC):
    """
    Prompt: 从【用户自然语言问题】中抽取【实体 + 关系】

    - 输入：一个自然语言问题（question）
    - 输出：该问题中显式或隐式涉及的【实体列表】和【关系列表】
    - 该 Prompt 用于 KG-RAG 中的 Query Understanding / Semantic Parsing 阶段
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a Knowledge Graph query understanding expert.

                === TASK ===
                Given a natural language QUESTION,
                extract the key KNOWLEDGE GRAPH ELEMENTS required to answer it.

                Specifically, you must extract:
                - ENTITIES explicitly mentioned in the question
                - RELATIONS (or attribute-type relations) implied or explicitly asked about

                === CORE OBJECTIVE ===
                Transform a natural language question into:
                - A set of entities (KG nodes)
                - A set of relations (KG edges or attribute relations)

                This output will be used for graph retrieval and reasoning.

                === STRICT EXTRACTION RULES ===

                ENTITIES:
                - Extract ONLY entities explicitly mentioned in the question
                - Do NOT invent entities
                - Do NOT include answer variables (e.g., date, city, organization)
                - Do NOT include generic types (e.g., "organization", "city")

                RELATIONS:
                - Extract canonical KG-style relations (verbs or predicates)
                - Relations may be:
                  - Explicit (e.g., "trained by" → trained_by)
                  - Implicit (multi-hop or compositional)
                - Attribute queries (e.g., release date) MUST be represented as relations

                MULTI-HOP QUESTIONS:
                - If the question implies multiple relational steps,
                  extract ALL required relations
                - Do NOT collapse multiple relations into one

                === FORBIDDEN ACTIONS ===
                ❌ Do NOT answer the question
                ❌ Do NOT generate triples
                ❌ Do NOT infer missing entities
                ❌ Do NOT use external knowledge
                ❌ Do NOT explain reasoning

                === OUTPUT FORMAT (STRICT JSON ONLY) ===
                {
                  "entities": ["Entity1", "Entity2"],
                  "relations": ["relation_1", "relation_2"]
                }

                Output JSON only. No extra text.
            """)
        else:
            return textwrap.dedent("""\
                你是一名【知识图谱问句理解】专家。

                === 任务 ===
                给定一个【用户自然语言问题】，
                抽取回答该问题所需的【知识图谱元素】。

                你必须抽取：
                - 问题中明确出现的【实体（Entities）】
                - 问题中显式或隐式表达的【关系（Relations）】

                该输出将用于后续的图谱检索与推理（KG-RAG）。

                === 核心目标 ===
                将自然语言问题转换为：
                - 实体集合（图谱节点）
                - 关系集合（图谱边 / 属性型关系）

                === 严格抽取规则 ===

                【实体】
                - 只抽取问题中【明确出现】的实体
                - 严禁臆造实体
                - 不抽取答案变量（如：日期、城市、组织）
                - 不抽取实体类型词（如“城市”“组织”“人”）

                【关系】
                - 抽取标准化的知识图谱关系表达
                - 可以是：
                  - 显式关系（如“trained by” → trained_by）
                  - 隐式关系（多跳 / 组合语义）
                - 属性查询（如发布日期）也必须建模为【关系】

                【多跳问题】
                - 若问题隐含多个关系路径，必须全部抽取
                - 不允许将多跳关系合并为单一关系

                === 严禁行为 ===
                ❌ 不回答问题
                ❌ 不生成三元组
                ❌ 不补全或猜测实体
                ❌ 不使用外部或常识知识
                ❌ 不输出推理过程

                === 输出格式（严格 JSON）===
                {
                  "entities": ["实体1", "实体2"],
                  "relations": ["关系1", "关系2"]
                }

                仅输出 JSON，不要输出任何多余文本。
            """)

    def build_prompt(self, question: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Question:
                {question}

                Extract entities and relations in JSON:
            """)
        else:
            return textwrap.dedent(f"""\
                用户问题：
                {question}

                请抽取实体与关系，并以 JSON 格式输出：
            """)
