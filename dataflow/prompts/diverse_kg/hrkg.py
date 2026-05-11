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

                A Hyper-Relation KG extends a standard Entity–Relation–Entity triple by attaching
                structured attributes to the RELATION. These attributes describe contextual constraints
                such as time, location, condition, reason, purpose, manner, degree, frequency, source,
                evidence, historical background, or other relation-specific information.

                === TASK DEFINITION ===
                Extract hyper-relation knowledge in the following string format:

                "<subj> Entity <obj> Entity <rel> Relation <semantic_attribute_name> attribute_value"

                === CORE RULES ===

                1. ENTITY:
                   - The subject and object must be clear nouns or noun phrases.
                   - Do NOT use pronouns such as he, she, it, they.
                   - Normalize entity names when possible.
                   - Keep entities concise.

                2. RELATION:
                   - The relation must describe the core fact between subject and object.
                   - Use concise semantic relation names such as:
                     BornIn, MarriedTo, Released, PerformedAt, Experienced, CausedBy,
                     InfluencedBy, StudiedAt, CollaboratedWith, Won, NominatedFor.
                   - Do NOT put time, location, reason, purpose, or manner inside the relation name.
                   - Each hyper-relation must express only ONE core fact.

                3. RELATION ATTRIBUTES:
                   - Attributes modify the RELATION, not the entity.
                   - Attribute names are NOT restricted to a fixed vocabulary.
                   - You may create any concise and meaningful semantic attribute name based on the text.
                   - Attribute names must clearly describe the role of the attribute value.
                   - Good attribute names include examples such as:
                     <time>, <location>, <reason>, <cause>, <purpose>, <manner>,
                     <condition>, <degree>, <frequency>, <source>, <evidence>,
                     <historical_context>, <political_context>, <award_year>,
                     <publication_date>, <performance_place>, <album_name>.
                   - Do NOT invent attributes or values.
                   - If no valid attribute is available, output only:
                     "<subj> Entity <obj> Entity <rel> Relation"

                4. FORBIDDEN ATTRIBUTE NAMES:
                   Never use placeholder or generic attribute names such as:
                   <attribute1>, <attribute2>, <attribute3>,
                   <attributeName1>, <attributeName2>,
                   <attr1>, <attr2>, <property1>, <property2>.

                   Incorrect:
                   "<subj> Chopin <obj> Financial Struggle <rel> Experienced <attribute1> political strife and instability <attribute2> time February 1848"

                   Correct:
                   "<subj> Chopin <obj> Financial Struggle <rel> Experienced <political_context> political strife and instability <time> February 1848"

                    Incorrect:
                    "<subj> Beyoncé Entity <obj> Destiny's Child Entity <rel> MemberOfRelation <time_period> late 1990s"

                    Correct:
                    "<subj> Beyoncé <obj> Destiny's Child <rel> MemberOf <time_period> late 1990s"

                    Incorrect:
                    "<subj> Beyoncé Entity <obj> Dangerously in Love Entity <rel> ReleasedRelation <time> 2003"

                    Correct:
                    "<subj> Beyoncé <obj> Dangerously in Love <rel> Released <time> 2003"

                5. ATTRIBUTE FORMAT:
                   - Every attribute must be written as an XML-like tag.
                   - The attribute name should be lowercase or snake_case.
                   - Attribute type words should not be placed inside the value.

                   Incorrect:
                   "<attribute2> time February 1848"

                   Correct:
                   "<time> February 1848"

                === OUTPUT FORMAT ===
                Output ONLY a valid JSON object:

                {
                  "tuple": [
                    "<subj> subject_name <obj> object_name <rel> relation_name <attribute_name> attribute_value",
                    "<subj> subject_name <obj> object_name <rel> relation_name <attribute_name> attribute_value"
                  ]
                }

                Do NOT output explanations.
                Do NOT output markdown.
                Do NOT output placeholder attribute names.
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的 Hyper-Relation 知识图谱抽取专家。

                Hyper-Relation 知识图谱是在传统“实体-关系-实体”三元组基础上，
                为【关系】附加结构化属性，用于刻画时间、地点、条件、原因、目的、方式、程度、频率、
                来源、证据、历史背景或其他与该关系相关的上下文信息。

                === 任务定义 ===
                从文本中抽取如下字符串格式的超关系知识：

                "<subj> 实体 <obj> 实体 <rel> 关系 <语义化属性名> 属性值"

                === 核心规则 ===

                1. 实体：
                   - 主语和宾语必须是清晰的名词或名词短语。
                   - 禁止使用代词。
                   - 实体名应规范、简洁。

                2. 关系：
                   - 关系必须描述主语和宾语之间的核心事实。
                   - 使用简洁的语义关系名，例如：
                     BornIn, MarriedTo, Released, PerformedAt, Experienced, CausedBy,
                     InfluencedBy, StudiedAt, CollaboratedWith, Won, NominatedFor。
                   - 不要把时间、地点、原因、目的、方式写进关系名。
                   - 每条 hyper-relation 只表达一个核心事实。

                3. 关系属性：
                   - 属性修饰的是【关系】，不是实体。
                   - 不限制具体属性类型。
                   - 可以根据文本自由生成简洁、明确、有语义的属性名。
                   - 属性名必须能够说明属性值在该关系中的作用。
                   - 可使用的属性名示例包括但不限于：
                     <time>, <location>, <reason>, <cause>, <purpose>, <manner>,
                     <condition>, <degree>, <frequency>, <source>, <evidence>,
                     <historical_context>, <political_context>, <award_year>,
                     <publication_date>, <performance_place>, <album_name>。
                   - 不允许虚构属性或属性值。
                   - 如果没有合适属性，可以只输出：
                     "<subj> 实体 <obj> 实体 <rel> 关系"

                4. 禁止使用占位符属性名：
                   严禁使用以下属性名：
                   <attribute1>, <attribute2>, <attribute3>,
                   <attributeName1>, <attributeName2>,
                   <attr1>, <attr2>, <property1>, <property2>。

                   错误：
                   "<subj> Chopin <obj> Financial Struggle <rel> Experienced <attribute1> political strife and instability <attribute2> time February 1848"

                   正确：
                   "<subj> Chopin <obj> Financial Struggle <rel> Experienced <political_context> political strife and instability <time> February 1848"

                5. 属性格式：
                   - 每个属性必须写成 XML 风格标签。
                   - 属性名建议使用小写或 snake_case。
                   - 属性类型不要混进属性值里。

                   错误：
                   "<attribute2> time February 1848"

                   正确：
                   "<time> February 1848"

                === 输出格式 ===
                只输出合法 JSON 对象：

                {
                  "tuple": [
                    "<subj> 实体 <obj> 实体 <rel> 关系 <语义化属性名> 属性值",
                    "<subj> 实体 <obj> 实体 <rel> 关系 <语义化属性名> 属性值"
                  ]
                }

                不输出解释。
                不输出 markdown。
                不输出占位符属性名。
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract Hyper-Relation Knowledge Graphs from the following text according to the rules above.

                Important:
                - Attribute names are free-form and not restricted to a fixed vocabulary.
                - However, every attribute name must be meaningful and semantic.
                - Do NOT use placeholder attribute names such as <attribute1>, <attribute2>, <attributeName1>, <attr1>.
                - Use lowercase or snake_case attribute tags.
                - Output ONLY valid JSON.

                Text:
                {text}

                Output:
                {{
                  "tuple": [
                    "<subj> subject_name <obj> object_name <rel> relation_name <attribute_name> attribute_value"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                请按照上述规则，从以下文本中抽取 Hyper-Relation 知识图谱。

                重要要求：
                - 属性名可以自由生成，不限制为固定属性集合。
                - 但是每个属性名必须有明确语义。
                - 禁止使用 <attribute1>, <attribute2>, <attributeName1>, <attr1> 等占位符属性名。
                - 属性名建议使用小写或 snake_case。
                - 只输出合法 JSON。

                文本：
                {text}

                输出：
                {{
                  "tuple": [
                    "<subj> 实体 <obj> 实体 <rel> 关系 <语义化属性名> 属性值"
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
                - Each question must be answerable using exactly ONE tuple.
                - The answer must come directly from that tuple.
                - The question may ask about the subject, object, relation, or explicit
                  relation attributes in the tuple.
                - Do not combine information from multiple tuples.
                - Do not introduce external or implicit knowledge.

                Core requirements:
                1. Preserve the tuple meaning and explicit qualifiers.
                2. Do not ignore relation attributes such as time, location, condition,
                   purpose, value, degree, market, method, reason, source, evidence,
                   or any other explicitly provided attribute.
                3. Do not invent missing attributes or values.
                4. Do not explain reasoning.
                5. Each tuple should generate as many high-quality and non-redundant QA pairs as possible.
                6. If a tuple contains multiple useful elements, generate separate QA pairs for them:
                   - subject-oriented questions
                   - object-oriented questions
                   - relation-oriented questions
                   - attribute-oriented questions
                   - qualifier-aware questions
                7. Avoid duplicate or near-duplicate questions.
                8. Avoid questions with the same answer and nearly identical meaning unless they ask from clearly different perspectives.
                9. Questions should be natural, fluent, and directly answerable from the tuple.

                Allowed one-hop question types:
                - Ask for the object given the subject and relation.
                - Ask for the subject given the object and relation.
                - Ask for the relation between subject and object.
                - Ask for an explicit attribute value, such as time, location, reason, purpose, method, degree, or condition.
                - Ask a qualifier-aware factual question that still depends on only one tuple.

                Forbidden question types:
                - Questions requiring more than one tuple.
                - Questions requiring external knowledge.
                - Questions that infer unstated facts.
                - Duplicate or near-duplicate paraphrases.
                - Questions whose answer is not explicitly present in the tuple.

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
            - 每个问题必须且只能由一条 tuple 直接回答。
            - 答案必须直接来自该 tuple。
            - 问题可以询问主体、客体、关系，或 tuple 中显式给出的关系属性。
            - 不允许跨 tuple 组合信息。
            - 不允许引入外部知识或隐含推断。

            核心要求：
            1. 保持 tuple 原始语义和显式限定条件。
            2. 不要忽略 time、location、condition、purpose、value、degree、market、method、
               reason、source、evidence 或其他显式给出的关系属性。
            3. 不要虚构缺失的属性或属性值。
            4. 不输出推理过程。
            5. 每条 tuple 应在保证质量和不重复的前提下，尽可能多地生成问答对。
            6. 如果一条 tuple 中包含多个可提问元素，应尽量分别生成不同角度的问题：
               - 围绕主体提问
               - 围绕客体提问
               - 围绕关系提问
               - 围绕属性提问
               - 结合限定条件提问
            7. 避免重复或高度相似的问题。
            8. 如果多个问题答案相同，只有在提问角度明显不同的情况下才保留。
            9. 问题表达要自然流畅，并且必须能从该 tuple 中直接回答。

            允许的一跳问题类型：
            - 已知主体和关系，询问客体。
            - 已知客体和关系，询问主体。
            - 询问主体和客体之间的关系。
            - 询问显式属性值，例如时间、地点、原因、目的、方式、程度、条件等。
            - 结合属性限定条件生成事实性问题，但仍然只能依赖一条 tuple。

            禁止的问题类型：
            - 需要多条 tuple 才能回答的问题。
            - 需要外部知识的问题。
            - 需要推断未明示事实的问题。
            - 重复或高度相似的改写问题。
            - 答案不在 tuple 中明确出现的问题。

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

                Generate as many high-quality and non-redundant one-hop QA pairs as possible.
                For each tuple, try to generate QA pairs from different valid perspectives:
                subject, object, relation, and explicit attributes.

                Hyper-relational tuples:
                {tuples}

                Output QA_pairs in JSON format only:
            """)

        return textwrap.dedent(f"""\
            请严格按照上述规则，从以下超关系 tuples 中生成一跳问答对。

            请在保证质量和不重复的前提下，尽可能多地生成一跳问答对。
            对于每条 tuple，请尽量从不同有效角度生成问题：
            主体、客体、关系，以及显式属性。

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