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
                <Attribute1> AttributeValue1
                <Attribute2> AttributeValue2
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
                     • Time (when)
                     • Location (where)
                     • Condition (under what condition)
                     • Purpose / Goal (why)
                     • Manner / Method (how)
                     • Degree / Intensity
                     • Frequency
                   - Do NOT invent attributes or values

                4. FACT CONSTRAINT:
                   - Each hyper-relation expresses ONE core fact
                   - Attributes only add constraints to that fact
                   - Avoid mixing multiple relations into one extraction

                === OUTPUT FORMAT ===
                - Output ONLY a JSON object
                - Key: "tuple"
                - Each item is a single string formatted exactly as:

                  "<subj> Entity <obj> Entity <rel> Relation <AttributeName1> ValueName1 <AttributeName2> ValueName2"

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
                - 每条为字符串，格式严格为：

                  "<subj> 实体 <obj> 实体 <rel> 关系 <AttributeName1> 属性值1"

                - 不输出任何解释性文本
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract Hyper-Relation Knowledge Graphs from the following text according to the rules above.

                Text:
                {text}

                Output ONLY JSON:
                {{
                  "tuple": [
                    "<subj> Entity <obj> Entity <rel> Relation <AttributeName1> ValueName1",
                    "<subj> Entity <obj> Entity <rel> Relation <AttributeName1> ValueName1"
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
                    "<subj> 实体 <obj> 实体 <rel> 关系 <AttributeName1> 属性值1",
                    "<subj> 实体 <obj> 实体 <rel> 关系 <AttributeName1> 属性值1"
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