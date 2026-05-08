import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json

@PROMPT_REGISTRY.register()
class HRKGHyperRelationExtractorPrompt(PromptABC):
    """
    从文本中抽取 Hyper-Relation Knowledge Graph（超关系知识图谱）
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in extracting Hyper-Relation Knowledge Graphs from natural language text.

                A Hyper-Relation KG extends a standard (Entity–Relation–Entity) triple by attaching
                structured attributes to the RELATION, capturing contextual constraints such as
                time, condition, purpose, manner, frequency, reason, or degree.

                === TASK DEFINITION ===
                Extract hyper-relation knowledge in the following form:

                <subj> EntityName
                <obj> EntityName
                <rel> Relation
                <attribute1> attributeValue1
                <attribute2> attributeValue2
                ...

                === CORE RULES ===
                1. ENTITY:
                   - Clear noun or noun phrase (concrete or abstract)
                   - NO pronouns (he / she / it / they)
                   - Normalized, concise wording

                2. RELATION:
                   - A commonsense or semantic relation describing what / why / how
                   - Examples: UsedFor, Causes, CapableOf, AtLocation, Helps, Makes, HasProperty

                3. RELATION ATTRIBUTES (CRITICAL):
                   - Attributes MODIFY THE RELATION, NOT THE ENTITY
                   - Attributes must be explicitly supported by the text
                   - Typical attribute types include (but are not limited to):
                     • time (when)
                     • location (where)
                     • condition (under what condition)
                     • purpose / goal (why)
                     • manner / method (how)
                     • degree / intensity
                     • frequency
                   - Do NOT invent attributes or values

                4. FACT CONSTRAINT:
                   - Each hyper-relation expresses ONE core fact
                   - Attributes only add constraints to that fact
                   - Avoid mixing multiple relations into one extraction

                === OUTPUT FORMAT ===
                - Output ONLY a JSON object
                - Key: "tuple"
                - Each item is a single string formatted exactly as:

                  "<subj> Entity <obj> Entity <rel> Relation <attributeName1> valueName1 <attributeName2> valueName2"

                  Replace attributeName1, attributeName2 with specific relation attributes

                - Do NOT add explanations or extra text
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的 Hyper-Relation 知识图谱抽取专家。

                Hyper-Relation 知识图谱是在传统“实体-关系-实体”三元组的基础上，
                为【关系】附加结构化属性，用于刻画时间、条件、目的、方式、频率等上下文约束。

                === 任务定义 ===
                从文本中抽取如下格式的超关系知识：

                <subj> 实体名
                <obj> 实体名
                <rel> 关系
                <属性1> 属性值1
                <属性2> 属性值2
                ...

                === 核心规则 ===
                1. 实体（Entity）：
                   - 清晰的名词或名词短语（具体或抽象）
                   - 禁止使用代词（他 / 她 / 它 / 他们）
                   - 表达应规范、简洁

                2. 关系（Relation）：
                   - 描述“做什么 / 为什么 / 如何”的语义关系
                   - 示例：用于、导致、能够、位于、帮助、使、具有属性

                3. 关系属性（关键要求）：
                   - 属性是【关系的修饰信息】，不是实体属性
                   - 属性和值必须能从文本中明确推导
                   - 常见属性类型包括（但不限于）：
                     · 时间
                     · 地点
                     · 条件
                     · 目的
                     · 方式
                     · 程度
                     · 频率
                   - 严禁虚构属性或属性值

                4. 事实约束：
                   - 每条 hyper-relation 仅表达一个核心事实
                   - 属性仅用于补充该关系的上下文约束
                   - 不得在一条中混合多个不同关系

                === 输出格式 ===
                - 仅输出 JSON 对象
                - 键为 "tuple"
                - 每条为字符串，格式为：

                  "<subj> 实体 <obj> 实体 <rel> 关系 <attributeName1> 属性值1"

                  其中attributeName1用具体的关系属性代替

                - 不输出任何解释性文本
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract Hyper-Relation Knowledge Graphs from the following text according to the rules above.

                Replace attributeName1 with specific relation attributes.

                Text:
                {text}

                Output ONLY JSON:
                {{
                  "tuple": [
                    "<subj> Entity <obj> Entity <rel> Relation <attributeName1> valueName1",
                    "<subj> Entity <obj> Entity <rel> Relation <attributeName1> valueName1"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                按照上述规则，从以下文本中抽取 Hyper-Relation 知识图谱：

                文本：
                {text}

                仅输出 JSON：
                {{
                  "tuple": [
                    "<subj> 实体 <obj> 实体 <rel> 关系 <attributeName1> 属性值1",
                    "<subj> 实体 <obj> 实体 <rel> 关系 <attributeName1> 属性值1"
                  ]
                }}
            """)


@PROMPT_REGISTRY.register()
class HRKGTripleCompletenessPrompt(PromptABC):
    """
    Evaluate the completeness of KG triples.

    Each triple is formatted as:
        "<subj> ... <obj> ... <rel> ... <attr1> ... <attr2> ..."

    The model should judge whether the triple contains all necessary
    information for the relation and its attributes.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名知识图谱三元组质量评估专家。
                你的任务是评估每个三元组的**完整性**。

                ### 判断标准
                - 三元组是否包含主体、客体和关系
                - 关系所需的关键属性是否齐全
                - 属性信息是否清晰且合理
                - 判断三元组是否缺失重要信息

                ### 输出格式
                仅返回 JSON：
                {
                    "completeness_scores": [float, float, ...]
                }

                每个三元组对应一个分数，范围 0-1：
                1 = 信息完整
                0.5 = 部分信息缺失
                0 = 信息严重缺失或无法理解

                不要输出任何解释。
            """)

        else:
            return textwrap.dedent("""\
                You are an expert in Knowledge Graph triple quality evaluation.
                Your task is to evaluate the **completeness** of each triple.

                ### Evaluation Criteria
                - Does the triple contain subject, object, and relation?
                - Are the key attributes for the relation present?
                - Are attribute values clear and reasonable?
                - Determine if the triple is missing important information.

                ### Output Format
                Return ONLY a JSON object:

                {
                    "completeness_scores": [float, float, ...]
                }

                Each score corresponds to one triple (0-1):
                1 = fully complete
                0.5 = partially complete
                0 = severely incomplete or unclear

                Do not output explanations.
            """)

    def build_prompt(self, triples: list) -> str:
        """
        Format triples for LLM evaluation.

        Args:
            triples (list): list of triple strings
        """

        triple_block = ""
        for idx, t in enumerate(triples):
            triple_block += f"ID {idx}: {t}\n"

        if self.lang == "zh":
            return f"""请评估以下知识图谱三元组的完整性。

            --- Triples ---
            {triple_block}

            请返回每个三元组的完整性得分（0-1），并严格按照 JSON 输出。"""

        else:
            return f"""Evaluate the completeness of the following KG triples.

            --- Triples ---
            {triple_block}

            Return ONLY a JSON object containing completeness scores for each triple (0-1)."""


@PROMPT_REGISTRY.register()
class HRKGTripleConsistencyPrompt(PromptABC):
    """
    Evaluate the consistency of KG triples.

    Each triple is formatted as:
        "<subj> ... <obj> ... <rel> ... <attr1> ... <attr2> ..."

    The model should judge whether the triple's attributes are
    logically consistent with each other and with the relation.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名知识图谱三元组质量评估专家。
                你的任务是评估每个三元组的**一致性**。

                ### 判断标准
                - 三元组的主体、客体和关系是否逻辑上协调
                - 关系的不同属性是否相互一致（例如时间、地点、数值等是否合理匹配）
                - 检查是否存在明显矛盾或冲突信息

                ### 输出格式
                仅返回 JSON：
                {
                    "consistency_scores": [float, float, ...]
                }

                每个三元组对应一个分数，范围 0-1：
                1 = 完全一致
                0.5 = 部分一致，有轻微矛盾
                0 = 严重不一致或属性冲突

                不要输出任何解释。
            """)

        else:
            return textwrap.dedent("""\
                You are an expert in Knowledge Graph triple quality evaluation.
                Your task is to evaluate the **consistency** of each triple.

                ### Evaluation Criteria
                - Check if the subject, object, and relation are logically coherent
                - Check if the relation's different attributes are consistent (e.g., time, location, values)
                - Detect any obvious contradictions or conflicts

                ### Output Format
                Return ONLY a JSON object:

                {
                    "consistency_scores": [float, float, ...]
                }

                Each score corresponds to one triple (0-1):
                1 = fully consistent
                0.5 = partially consistent, minor conflicts
                0 = severely inconsistent or contradictory

                Do not output explanations.
            """)

    def build_prompt(self, triples: list) -> str:
        """
        Format triples for LLM evaluation.

        Args:
            triples (list): list of triple strings
        """

        triple_block = ""
        for idx, t in enumerate(triples):
            triple_block += f"ID {idx}: {t}\n"

        if self.lang == "zh":
            return f"""请评估以下知识图谱三元组的属性一致性。

            --- Triples ---
            {triple_block}

            请返回每个三元组的一致性得分（0-1），并严格按照 JSON 输出。"""

        else:
            return f"""Evaluate the consistency of the following KG triples.

            --- Triples ---
            {triple_block}

            Return ONLY a JSON object containing consistency scores for each triple (0-1)."""


@PROMPT_REGISTRY.register()
class HRKGOneHopQAPathGenerationPrompt(PromptABC):
    """
    Generate one-hop QA pairs from hyper-relational tuples.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a hyper-relational knowledge graph question-answer generation expert.

                Your task:
                Generate ONE-HOP question-answer pairs strictly based on the given
                hyper-relational tuples.

                Definition of ONE-HOP QA:
                - Each question must be answerable using exactly ONE tuple
                - The answer must come directly from that tuple
                - The question may ask about the subject, object, or explicit
                  relation attributes in the tuple
                - Do not combine information from multiple tuples
                - Do not introduce external or implicit knowledge

                Rules:
                - Preserve the tuple meaning and explicit qualifiers
                - Do not ignore relation attributes such as Time, Location,
                  Condition, Purpose, Value, Degree, Market, or Method
                - Do not invent missing attributes or values
                - Do not explain reasoning
                - Each tuple may generate one or more QA pairs
                - Questions should be natural and fluent

                Output format (STRICT JSON):
                    {
                    "QA_pairs": [
                        {
                        "question": "...",
                        "answer": "..."
                        }
                    ]
                    }
            """)

        return textwrap.dedent("""\
            你是超关系知识图谱问答生成专家。

            你的任务：
            严格基于给定的 hyper-relation tuples 生成一跳问答对。

            一跳 QA 定义：
            - 每个问题必须且只能由一条 tuple 直接回答
            - 答案必须直接来自该 tuple
            - 问题可以询问主体、客体或 tuple 中显式给出的关系属性
            - 不允许跨 tuple 组合信息
            - 不允许引入外部知识或隐含推断

            规则：
            - 保持 tuple 原始语义和限定条件
            - 不要忽略 Time、Location、Condition、Purpose、Value、Degree、Market、Method 等关系属性
            - 不要虚构缺失的属性或属性值
            - 不输出推理过程
            - 每条 tuple 可以生成一个或多个问答对
            - 问题表达要自然流畅

            输出格式（严格 JSON）：
            {
            "QA_pairs": [
                {
                "question": "...",
                "answer": "..."
                }
            ]
            }
        """)

    def build_prompt(self, tuples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate one-hop QA pairs strictly following the rules above.

                Hyper-relational tuples:
                {tuples}

                Output QA_pairs in JSON format only:
            """)

        return textwrap.dedent(f"""\
            请严格按照上述规则，从以下超关系 tuples 中生成一跳问答对。

            超关系 tuples：
            {tuples}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class HRKGTwoHopPathQAGenerationPrompt(PromptABC):
    """
    Generate two-hop QA pairs from hyper-relational paths.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a hyper-relational multi-hop question-answer generation expert.

                Your task:
                Generate QUESTION-ANSWER pairs that require EXACTLY TWO HOPS of reasoning,
                strictly based on the given two-hop hyper-relational paths.

                Critical requirements:
                1. Each QA must require both tuples in the path to answer.
                2. Do not generate one-hop questions.
                3. Relation attributes may be used as qualifiers in the question
                   or answer, but the QA must still depend on both hops.
                4. Do not introduce external knowledge or assumptions.
                5. Do not modify entity names, relation meaning, or attribute values.

                Allowed question patterns:
                - Two-step entity relation inference
                - Questions that use the first hop to identify an entity and the
                  second hop to obtain the answer
                - Questions that use explicit attributes as qualifiers in a
                  two-hop reasoning chain

                Forbidden question patterns:
                - Any question answerable from only one tuple
                - Direct one-hop subject-object questions
                - Questions that ignore the path connection

                Output format (STRICT JSON):
            {
            "QA_pairs": [
                {
                "question": "...",
                "answer": "..."
                }
            ]
            }
            """)

        return textwrap.dedent("""\
            你是超关系多跳知识图谱问答生成专家。

            你的任务：
            严格基于给定的两跳超关系路径生成问答对。

            关键要求：
            1. 每个 QA 必须依赖路径中的两条 tuple 才能回答。
            2. 不允许生成一跳问题。
            3. 关系属性可以作为问题或答案中的限定条件，但 QA 仍必须依赖两跳。
            4. 不允许引入外部知识或隐含假设。
            5. 不允许修改实体名、关系语义或属性值。

            允许的问题类型：
            - 两步实体关系推理
            - 先由第一跳定位实体，再由第二跳得到答案的问题
            - 使用显式属性作为限定条件的两跳推理问题

            禁止的问题类型：
            - 只依赖一条 tuple 就能回答的问题
            - 直接的一跳主客体问题
            - 无视路径连接关系的问题

            输出格式（严格 JSON）：
            {
            "QA_pairs": [
                {
                "question": "...",
                "answer": "..."
                }
            ]
            }
        """)

    def build_prompt(self, paths: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate two-hop QA pairs strictly following the rules above.

                Two-hop hyper-relational paths:
                {paths}

                Output QA_pairs in JSON format only:
            """)

        return textwrap.dedent(f"""\
            请严格按照上述规则，从以下两跳超关系路径中生成问答对。

            两跳超关系路径：
            {paths}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class HRKGRelationTripleSubgraphNumericQAPrompt(PromptABC):
    """
    Generate numeric QA pairs from a hyper-relational subgraph.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a hyper-relational knowledge graph QA generation expert.

                === TASK ===
                Given a subgraph composed of hyper-relational tuples, generate
                numeric QA pairs.

                === CORE REQUIREMENTS ===
                1. The answer must be a NUMBER.
                2. Each question must rely on at least two tuples.
                3. You may use explicit relation attributes such as Time,
                   Location, Condition, Purpose, Value, Degree, Market, Method,
                   Capacity, or Frequency when forming the question.
                4. Use only the given tuples; do not introduce external knowledge.
                5. Do not ignore explicit qualifiers in the tuples.

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do not explain reasoning or mention tuples explicitly.
            """)

        return textwrap.dedent("""\
            你是超关系知识图谱数值型问答生成专家。

            === 任务 ===
            给定由 hyper-relation tuples 构成的子图，生成数值型 QA。

            === 核心要求 ===
            1. 答案必须是数字。
            2. 每个问题必须依赖至少两条 tuple。
            3. 可以使用 Time、Location、Condition、Purpose、Value、Degree、Market、Method、Capacity、Frequency 等显式关系属性构造问题。
            4. 只能使用给定 tuples，不允许引入外部知识。
            5. 不要忽略 tuple 中显式给出的限定条件。

            === 输出格式 ===
            {
              "QA_pairs": [
                {
                  "question": "...",
                  "answer": "..."
                }
              ]
            }

            不输出推理过程，也不要直接提及 tuples 本身。
        """)

    def build_prompt(self, tuples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate numeric QA pairs strictly following the rules above.

                Each question must rely on at least two tuples.

                Hyper-relational subgraph tuples:
                {tuples}

                Output QA pairs in JSON format only:
            """)

        return textwrap.dedent(f"""\
            请严格按照上述规则，从以下超关系子图 tuples 中生成数值型 QA。

            每个问题必须依赖至少两条 tuple。

            超关系子图 tuples：
            {tuples}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class HRKGRelationTripleSubgraphSetQAPrompt(PromptABC):
    """
    Generate set-based QA pairs from a hyper-relational subgraph.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a hyper-relational knowledge graph QA generation expert.

                === TASK ===
                Given a subgraph composed of hyper-relational tuples, generate
                set-based QA pairs.

                === CORE REQUIREMENTS ===
                1. The answer must be a SET, such as a comma-separated list of entities,
                   concepts, values, locations, or other explicit tuple results.
                2. Each question must rely on at least two tuples.
                3. You may use explicit relation attributes such as Time,
                   Location, Condition, Purpose, Value, Degree, Market, Method,
                   Capacity, or Frequency when forming the question.
                4. Use only the given tuples; do not introduce external knowledge.
                5. Do not ignore explicit qualifiers in the tuples.

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do not explain reasoning or mention tuples explicitly.
                Ensure answers are clear set-like outputs.
            """)

        return textwrap.dedent("""\
            你是超关系知识图谱集合型问答生成专家。

            === 任务 ===
            给定由 hyper-relation tuples 构成的子图，生成集合型 QA。

            === 核心要求 ===
            1. 答案必须是集合形式，例如由逗号分隔的实体、概念、数值、地点或其他显式结果。
            2. 每个问题必须依赖至少两条 tuple。
            3. 可以使用 Time、Location、Condition、Purpose、Value、Degree、Market、Method、Capacity、Frequency 等显式关系属性构造问题。
            4. 只能使用给定 tuples，不允许引入外部知识。
            5. 不要忽略 tuple 中显式给出的限定条件。

            === 输出格式 ===
            {
              "QA_pairs": [
                {
                  "question": "...",
                  "answer": "..."
                }
              ]
            }

            不输出推理过程，也不要直接提及 tuples 本身。
            确保答案是清晰的集合形式。
        """)

    def build_prompt(self, tuples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate set-based QA pairs strictly following the rules above.

                Each question must rely on at least two tuples.

                Hyper-relational subgraph tuples:
                {tuples}

                Output QA pairs in JSON format only:
            """)

        return textwrap.dedent(f"""\
            请严格按照上述规则，从以下超关系子图 tuples 中生成集合型 QA。

            每个问题必须依赖至少两条 tuple。

            超关系子图 tuples：
            {tuples}

            仅以 JSON 格式输出 QA_pairs：
        """)