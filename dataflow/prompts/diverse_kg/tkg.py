import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class TKGRelationQuadrupleExtractorPrompt(PromptABC):
    """
    Extract temporal relation quadruples from text:
    <subj> Entity <obj> Entity <rel> Relation <time> TimeValue
    If no time is mentioned, fill <time> with 'NA'.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in extracting temporal relation quadruples from natural language text.

                Each quadruple MUST follow this exact format:
                <subj> Entity <obj> Entity <rel> Relation <time> TimeValue

                === TIME STANDARDIZATION ===
                1. Specific date: YYYY-MM-DD, e.g., 2025-03-03
                2. Month: Full month name + year, e.g., March 2025
                3. Year: YYYY, e.g., 2025
                4. Quarter: QX YYYY, e.g., Q1 2025
                5. Time span / interval: start_date|end_date, e.g., 2025-01-01|2025-01-03
                6. If no time is explicitly mentioned in the text, set <time> to 'NA'.

                === CORE RULES ===
                - ENTITY: clear noun/noun phrase, no pronouns
                - RELATION: semantic relation describing what/why/how
                - TIME: optional; fill with standardized value if present, otherwise 'NA'
                - Each quadruple expresses ONE core fact
                - Do NOT invent entities, relations, or times beyond the text

                === OUTPUT FORMAT ===
                - JSON object only
                - Key: "tuple"
                - Example items:

                  "<subj> Entity <obj> Entity <rel> Relation <time> 2025-03-03"
                  "<subj> Entity <obj> Entity <rel> Relation <time> March 2025"
                  "<subj> Entity <obj> Entity <rel> Relation <time> 2025"
                  "<subj> Entity <obj> Entity <rel> Relation <time> Q1 2025"
                  "<subj> Entity <obj> Entity <rel> Relation <time> 2025-01-01|2025-01-03"
                  "<subj> Entity <obj> Entity <rel> Relation <time> NA"

                - Do NOT add explanations or extra text
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的时间关系四元组抽取专家。

                每条四元组必须严格遵循：
                <subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值

                === 时间标准化 ===
                1. 具体日期：YYYY-MM-DD，例如 2025-03-03
                2. 月份：完整月份名称 + 年份，例如 March 2025
                3. 年份：YYYY，例如 2025
                4. 季度：QX YYYY，例如 Q1 2025
                5. 时间段/区间：起始日期|结束日期，例如 2025-01-01|2025-01-03
                6. 如果文本中没有明确时间，<time> 填入 'NA'

                === 核心规则 ===
                - 实体：清晰名词/名词短语，不使用代词
                - 关系：描述“做什么/为什么/如何”的语义关系
                - 时间：可选；如果文本明确，使用标准化值，否则填 'NA'
                - 每条四元组仅表达一个核心事实
                - 严禁虚构实体、关系或时间

                === 输出格式 ===
                - 仅输出 JSON 对象
                - 键为 "tuple"
                - 示例条目：

                  "<subj> 实体 <obj> 实体 <rel> 关系 <time> 2025-03-03"
                  "<subj> 实体 <obj> 实体 <rel> 关系 <time> March 2025"
                  "<subj> 实体 <obj> 实体 <rel> 关系 <time> 2025"
                  "<subj> 实体 <obj> 实体 <rel> 关系 <time> Q1 2025"
                  "<subj> 实体 <obj> 实体 <rel> 关系 <time> 2025-01-01|2025-01-03"
                  "<subj> 实体 <obj> 实体 <rel> 关系 <time> NA"

                - 不输出任何解释性文本
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract temporal relation quadruples from the following text according to the rules above.

                Text:
                {text}

                Output ONLY JSON:
                {{
                  "tuple": [
                    "<subj> Entity <obj> Entity <rel> Relation <time> TimeValue"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                按照上述规则，从以下文本中抽取时间关系四元组：

                文本：
                {text}

                仅输出 JSON：
                {{
                  "tuple": [
                    "<subj> 实体 <obj> 实体 <rel> 关系 <time> 时间值"
                  ]
                }}
            """)


@PROMPT_REGISTRY.register()
class TKGAttributeQuadrupleExtractorPrompt(PromptABC):
    """
    Extract temporal entity-attribute quadruples from text:
    <entity> Entity <attribute> AttributeName <value> AttributeValue <time> TimeValue
    If no time is mentioned, fill <time> with 'NA'.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in extracting temporal entity-attribute quadruples from natural language text.

                Each quadruple MUST follow this exact format:
                <entity> Entity <attribute> AttributeName <value> AttributeValue <time> TimeValue

                === TIME STANDARDIZATION ===
                1. Specific date: YYYY-MM-DD, e.g., 2025-03-03
                2. Month: Full month name + year, e.g., March 2025
                3. Year: YYYY, e.g., 2025
                4. Quarter: QX YYYY, e.g., Q1 2025
                5. Time span / interval: start_date|end_date, e.g., 2025-01-01|2025-01-03
                6. If no time is explicitly mentioned in the text, set <time> to 'NA'.

                === CORE RULES ===
                - ENTITY: clear noun/noun phrase, no pronouns
                - ATTRIBUTE: the property of the entity explicitly stated in the text
                - VALUE: the value corresponding to the attribute
                - TIME: optional; fill with standardized value if present, otherwise 'NA'
                - Each quadruple expresses ONE fact
                - Do NOT invent entities, attributes, values, or times beyond the text

                === OUTPUT FORMAT ===
                - JSON object only
                - Key: "tuple"
                - Example items:

                  "<entity> Henry <attribute> profession <value> musician <time> 2025-03-03"
                  "<entity> Henry <attribute> profession <value> musician <time> March 2025"
                  "<entity> Henry <attribute> profession <value> musician <time> 2025"
                  "<entity> Henry <attribute> profession <value> musician <time> Q1 2025"
                  "<entity> Henry <attribute> profession <value> musician <time> 2025-01-01|2025-01-03"
                  "<entity> Henry <attribute> profession <value> musician <time> NA"

                - Do NOT add explanations or extra text
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的时间实体属性四元组抽取专家。

                每条四元组必须严格遵循：
                <entity> 实体 <attribute> 属性名 <value> 属性值 <time> 时间值

                === 时间标准化 ===
                1. 具体日期：YYYY-MM-DD，例如 2025-03-03
                2. 月份：完整月份名称 + 年份，例如 March 2025
                3. 年份：YYYY，例如 2025
                4. 季度：QX YYYY，例如 Q1 2025
                5. 时间段/区间：起始日期|结束日期，例如 2025-01-01|2025-01-03
                6. 如果文本中没有明确时间，<time> 填入 'NA'

                === 核心规则 ===
                - 实体：清晰名词/名词短语，不使用代词
                - 属性：文本明确表述的实体属性
                - 值：对应属性的值
                - 时间：可选；如果文本明确，使用标准化值，否则填 'NA'
                - 每条四元组仅表达一个事实
                - 严禁虚构实体、属性、值或时间

                === 输出格式 ===
                - 仅输出 JSON 对象
                - 键为 "tuple"
                - 示例条目：

                  "<entity> Henry <attribute> profession <value> musician <time> 2025-03-03"
                  "<entity> Henry <attribute> profession <value> musician <time> March 2025"
                  "<entity> Henry <attribute> profession <value> musician <time> 2025"
                  "<entity> Henry <attribute> profession <value> musician <time> Q1 2025"
                  "<entity> Henry <attribute> profession <value> musician <time> 2025-01-01|2025-01-03"
                  "<entity> Henry <attribute> profession <value> musician <time> NA"

                - 不输出任何解释性文本
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract temporal entity-attribute quadruples from the following text according to the rules above.

                Text:
                {text}

                Output ONLY JSON:
                {{
                  "tuple": [
                    "<entity> Entity <attribute> AttributeName <value> AttributeValue <time> TimeValue"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                按照上述规则，从以下文本中抽取时间实体属性四元组：

                文本：
                {text}

                仅输出 JSON：
                {{
                  "tuple": [
                    "<entity> 实体 <attribute> 属性名 <value> 属性值 <time> 时间值"
                  ]
                }}
            """)


@PROMPT_REGISTRY.register()
class TKGAttributeTimePointQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从【实体-关系-属性值-时间】四元组生成【时序点型 QA】

    - 问题针对具体时间
    - 答案为四元组中的时间值
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating temporal knowledge graph QA.

                === TASK ===
                Given:
                - ENTITY–ATTRIBUTE–VALUE–TIME quadruples

                Generate QA pairs such that:

                === CORE REQUIREMENT ===
                - Each question asks for the **specific time** associated with an entity's attribute
                - The answer MUST be the time value from the quadruple
                - Do NOT invent times or entities beyond the given quadruples

                Each QA MUST:
                - Mention the entity and attribute explicitly in the question
                - Ask something like:
                  "When did [Entity] have [Attribute] value [Value]?"
                  or
                  "At what time was [Attribute] of [Entity] equal to [Value]?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "TimeValue"
                    }
                  ]
                }

                Do NOT explain reasoning or mention quadruples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【带时间属性四元组】问答生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值-时间】四元组

                目标：
                生成 QA 对，要求：

                === 核心要求 ===
                - 问题围绕具体时间
                - 答案为四元组中的时间值
                - 不得虚构时间或实体

                每条 QA 必须：
                - 在问题中明确提及实体和属性
                - 问句示例：
                  “[实体] 的 [属性] 值为 [属性值] 的时间是什么？”
                  “[属性] 为 [属性值] 的 [实体] 发生在什么时候？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "时间值"
                    }
                  ]
                }

                不输出推理过程，不提及四元组本身。
            """)

    def build_prompt(self, temporal_quadruples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate **temporal QA pairs** strictly following the rules above.

                ENTITY–ATTRIBUTE–VALUE–TIME quadruples:
                {temporal_quadruples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下带时间的实体属性四元组中生成【时序点型 QA】。

                实体-属性-属性值-时间四元组：
                {temporal_quadruples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class TKGAttributeEventOrderQAGenerationPrompt(PromptABC):
    """
    从【实体-属性-属性值-时间】四元组生成【大量事件顺序型 QA】

    - 基于时间排序
    - 对所有可比较事件生成问答
    - 生成“之后做了什么”的问题
    - 答案为后一事件（属性+属性值）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating LARGE-SCALE temporal event-order QA.

                === TASK ===
                Given:
                - ENTITY–ATTRIBUTE–VALUE–TIME quadruples

                You MUST:

                1. Group events by entity
                2. Sort events chronologically for each entity
                3. Use ALL valid event pairs where one event occurs earlier than another
                4. Generate QA pairs asking what happened AFTER the earlier event
                5. For EACH valid pair, generate MULTIPLE different question forms

                === CORE REQUIREMENTS ===
                - Use as many valid event pairs as possible
                - Do NOT generate only one QA
                - For each event pair, produce at least 2–3 distinct question styles
                - Questions must not be identical
                - Question must clearly describe the earlier event
                - Do NOT invent events or times

                === QUESTION STYLE VARIATIONS ===
                - "After [Event A], what happened next?"
                - "What did [Entity] do after [Event A]?"
                - "Following [Event A], what occurred?"
                - "What was the next event after [Event A]?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Generate as many valid QA pairs as possible.
                Do NOT explain reasoning.
                Do NOT mention quadruples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【大规模时间顺序事件问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值-时间】四元组

                你必须：

                1. 按实体分组
                2. 对每个实体的事件按时间排序
                3. 使用所有时间靠前→时间靠后的事件组合
                4. 针对每一对事件生成“之后发生了什么”的问答
                5. 每一对事件至少生成 2–3 种不同表达方式

                === 核心要求 ===
                - 使用尽可能多的有效事件组合
                - 不得只生成少量问答
                - 问题必须明确描述“前一个事件”
                - 答案必须是“后一个事件”（属性 + 属性值）
                - 不得虚构事件或时间
                - 问句不得重复

                === 问法示例（需多样化表达）===
                - “[实体] 在 [属性] 为 [属性值] 之后做了什么？”
                - “当 [实体] 的 [属性] 为 [属性值] 后，接下来发生了什么？”
                - “[事件A] 之后，[实体] 的下一件事情是什么？”
                - “在完成 [事件A] 后，[实体] 又发生了什么？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                尽可能生成最多的 QA 对。
                不输出推理过程，不提及四元组本身。
            """)

    def build_prompt(self, temporal_quadruples: str):
        return textwrap.dedent(f"""\
            请严格按照规则，从以下带时间的实体属性四元组中生成【尽可能多的事件顺序型 QA】。

            实体-属性-属性值-时间四元组：
            {temporal_quadruples}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class TKGAttributeTimeOrderQAGenerationPrompt(PromptABC):
    """
    从【实体-属性-属性值-时间】四元组生成大量【时间先后比较型 QA】

    - 对同一实体的所有事件两两组合
    - 每对生成多种表达方式
    - 答案为时间更早的事件（属性+属性值）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating LARGE-SCALE temporal comparison QA.

                === TASK ===
                Given:
                - ENTITY–ATTRIBUTE–VALUE–TIME quadruples

                You must:
                1. Group events by entity
                2. For each entity, compare ALL possible pairs of events
                3. For EACH pair, generate MULTIPLE distinct question forms
                4. Determine which event happened earlier

                === CORE REQUIREMENTS ===
                - Use all valid event pairs
                - Each pair must produce at least 3 different question styles
                - Questions must not be identical
                - Mention both events explicitly
                - Do NOT invent events or times

                === QUESTION STYLE VARIATIONS ===
                - Direct comparison:
                  "Which happened earlier: A or B?"
                - Choice form:
                  "Did E do A before B?"
                - Explicit time reasoning:
                  "Between A and B, which occurred first?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Generate as many valid QA pairs as possible.
                Do NOT explain reasoning.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【大规模时间先后比较问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值-时间】四元组

                你必须：

                1. 按实体分组
                2. 对每个实体的所有事件做两两组合
                3. 每一对事件至少生成 3 种不同问法
                4. 判断哪个事件更早

                === 核心要求 ===
                - 使用所有可比较事件对
                - 每对至少生成 3 个不同表达问题
                - 问题不得重复
                - 问题中必须明确提及两个事件
                - 不得虚构事件或时间

                === 问法类型示例 ===
                - 直接比较型：
                  “[事件A] 和 [事件B] 哪个更早？”
                - 选择型：
                  “[实体] 是先 [事件A] 还是先 [事件B]？”
                - 判断型：
                  “[实体] 是否在 [事件B] 之前完成 [事件A]？”
                - 比较型：
                  “在 [事件A] 和 [事件B] 中，哪一个发生得更早？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                尽可能生成最多的 QA 对。
                不输出推理过程。
            """)

    def build_prompt(self, temporal_quadruples: str):
        return textwrap.dedent(f"""\
            请严格按照规则，从以下四元组生成【尽可能多的时间先后比较型 QA】。

            实体-属性-属性值-时间四元组：
            {temporal_quadruples}

            仅输出 JSON 格式 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class TKGAttributeTimeIntervalQAGenerationPrompt(PromptABC):
    """
    从【实体-属性-属性值-时间】四元组生成【大规模时间区间型 QA】

    - 基于时间排序
    - 构造时间区间
    - 生成“在某时间区间内发生了什么”的问答
    - 答案为区间内的事件（属性+属性值）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating LARGE-SCALE temporal interval QA.

                === TASK ===
                Given:
                - ENTITY–ATTRIBUTE–VALUE–TIME quadruples

                You MUST:

                1. Group events by entity
                2. Sort events chronologically
                3. Construct ALL valid time intervals using earlier and later events
                4. Generate QA asking what happened BETWEEN two times or two events
                5. Use as many valid interval combinations as possible
                6. For each interval, generate multiple question variations

                === CORE REQUIREMENTS ===
                - Use ALL possible valid time intervals
                - Do NOT generate only a few QA
                - Each question must clearly define a time interval
                - The answer must list the event(s) inside the interval
                - Do NOT invent events or times
                - Questions must not repeat

                === QUESTION STYLE VARIATIONS ===
                - "What happened between [Time A] and [Time B]?"
                - "What did [Entity] do between [Event A] and [Event B]?"
                - "During the period from [Time A] to [Time B], what occurred?"
                - "After [Event A] but before [Event B], what happened?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "EventDescription or ListOfEvents"
                    }
                  ]
                }

                Generate as many valid QA pairs as possible.
                Do NOT explain reasoning.
                Do NOT mention quadruples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【大规模时间区间问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值-时间】四元组

                你必须：

                1. 按实体分组
                2. 按时间排序
                3. 构造所有合法的时间区间（起始时间 < 结束时间）
                4. 针对每个区间生成“在这段时间内发生了什么”的问答
                5. 每个区间至少生成 2–3 种不同表达方式
                6. 尽可能使用全部区间组合

                === 核心要求 ===
                - 使用尽可能多的区间组合
                - 不得只生成少量问答
                - 问题必须明确给出时间区间或区间边界事件
                - 答案必须是该区间内真实存在的事件（属性 + 属性值）
                - 不得虚构事件或时间
                - 问句不得重复

                === 问法示例（需多样化表达）===
                - “[时间A] 到 [时间B] 之间发生了什么？”
                - “[实体] 在 [时间A] 至 [时间B] 期间做了什么？”
                - “在完成 [事件A] 之后、发生 [事件B] 之前发生了什么？”
                - “[事件A] 和 [事件B] 之间发生了哪些事情？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "区间内事件描述（单个或多个）"
                    }
                  ]
                }

                尽可能生成最多的 QA 对。
                不输出推理过程，不提及四元组本身。
            """)

    def build_prompt(self, temporal_quadruples: str):
        return textwrap.dedent(f"""\
            请严格按照规则，从以下带时间的实体属性四元组中生成【尽可能多的时间区间型 QA】。

            实体-属性-属性值-时间四元组：
            {temporal_quadruples}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class TKGTupleTimePointQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从【实体-关系-实体-时间】四元组生成【时序点型 QA】

    - 问题针对具体时间
    - 答案为四元组中的时间值
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating temporal knowledge graph QA.

                === TASK ===
                Given:
                - ENTITY–RELATION–ENTITY–TIME quadruples

                Generate QA pairs such that:

                === CORE REQUIREMENT ===
                - Each question asks for the **specific time** associated with an entity relation
                - The answer MUST be the time value from the quadruple
                - Do NOT invent times, entities, or relations beyond the given quadruples

                Each QA MUST:
                - Mention the two entities and the relation explicitly in the question
                - Ask something like:
                  "When did [Entity1] [Relation] [Entity2]?"
                  or
                  "At what time did [Relation] occur between [Entity1] and [Entity2]?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "TimeValue"
                    }
                  ]
                }

                Do NOT explain reasoning or mention quadruples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【带时间实体关系四元组】问答生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体-时间】四元组

                目标：
                生成 QA 对，要求：

                === 核心要求 ===
                - 问题围绕具体时间
                - 答案为四元组中的时间值
                - 不得虚构时间、实体或关系

                每条 QA 必须：
                - 在问题中明确提及两个实体和关系
                - 问句示例：
                  “[实体1] 在什么时候 [关系] [实体2]？”
                  “[关系] 发生在 [实体1] 和 [实体2] 之间的时间是什么？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "时间值"
                    }
                  ]
                }

                不输出推理过程，不提及四元组本身。
            """)

    def build_prompt(self, temporal_quadruples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate **temporal QA pairs** strictly following the rules above.

                ENTITY–RELATION–ENTITY–TIME quadruples:
                {temporal_quadruples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下带时间的实体关系四元组中生成【时序点型 QA】。

                实体-关系-实体-时间四元组：
                {temporal_quadruples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class TKGTupleEventOrderQAGenerationPrompt(PromptABC):
    """
    从【实体-关系-实体-时间】四元组生成【大量事件顺序型 QA】

    - 基于时间排序
    - 对所有可比较事件生成问答
    - 生成“之后发生了什么”的问题
    - 答案为后一事件（关系+实体）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating LARGE-SCALE temporal event-order QA.

                === TASK ===
                Given:
                - ENTITY–RELATION–ENTITY–TIME quadruples

                You MUST:

                1. Group events by the first entity
                2. Sort events chronologically for each entity
                3. Use ALL valid event pairs where one event occurs earlier than another
                4. Generate QA pairs asking what happened AFTER the earlier event
                5. For EACH valid pair, generate MULTIPLE different question forms

                === CORE REQUIREMENTS ===
                - Use as many valid event pairs as possible
                - Do NOT generate only one QA
                - For each event pair, produce at least 2–3 distinct question styles
                - Questions must not be identical
                - Question must clearly describe the earlier event
                - Do NOT invent events or times

                === QUESTION STYLE VARIATIONS ===
                - "After [Entity1] [Relation] [Entity2], what happened next?"
                - "What did [Entity1] do after [Relation] with [Entity2]?"
                - "Following [Entity1] [Relation] [Entity2], what occurred?"
                - "What was the next event after [Entity1] [Relation] [Entity2]?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Generate as many valid QA pairs as possible.
                Do NOT explain reasoning.
                Do NOT mention quadruples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【大规模时间顺序事件问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体-时间】四元组

                你必须：

                1. 按第一个实体分组
                2. 对每个实体的事件按时间排序
                3. 使用所有时间靠前→时间靠后的事件组合
                4. 针对每一对事件生成“之后发生了什么”的问答
                5. 每一对事件至少生成 2–3 种不同表达方式

                === 核心要求 ===
                - 使用尽可能多的有效事件组合
                - 不得只生成少量问答
                - 问题必须明确描述“前一个事件”
                - 答案必须是“后一个事件”（关系 + 实体）
                - 不得虚构事件或时间
                - 问句不得重复

                === 问法示例（需多样化表达）===
                - “[实体1] 在 [关系] [实体2] 之后做了什么？”
                - “当 [实体1] 的 [关系] 为 [实体2] 后，接下来发生了什么？”
                - “[前事件] 之后，[实体1] 的下一件事情是什么？”
                - “在完成 [前事件] 后，[实体1] 又发生了什么？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "后一个事件（关系+实体）"
                    }
                  ]
                }

                尽可能生成最多的 QA 对。
                不输出推理过程，不提及四元组本身。
            """)

    def build_prompt(self, temporal_quadruples: str):
        return textwrap.dedent(f"""\
            请严格按照规则，从以下带时间的实体关系四元组中生成【尽可能多的事件顺序型 QA】。

            实体-关系-实体-时间四元组：
            {temporal_quadruples}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class TKGTupleTimeOrderQAGenerationPrompt(PromptABC):
    """
    从【实体-关系-实体-时间】四元组生成大量【时间先后比较型 QA】

    - 对同一实体的所有事件两两组合
    - 每对生成多种表达方式
    - 答案为时间更早的事件（关系+实体）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating LARGE-SCALE temporal comparison QA.

                === TASK ===
                Given:
                - ENTITY–RELATION–ENTITY–TIME quadruples

                You must:
                1. Group events by the first entity
                2. For each entity, compare ALL possible pairs of events
                3. For EACH pair, generate MULTIPLE distinct question forms
                4. Determine which event happened earlier

                === CORE REQUIREMENTS ===
                - Use all valid event pairs
                - Each pair must produce at least 3 different question styles
                - Questions must not be identical
                - Mention both events explicitly
                - Do NOT invent events or times

                === QUESTION STYLE VARIATIONS ===
                - Direct comparison:
                  "Which happened earlier: [Entity1] [Relation] [Entity2] or [Entity1] [Relation] [Entity2]?"
                - Choice form:
                  "Did [Entity1] do [Relation] [Entity2] before [Relation] [Entity2]?"
                - Explicit time reasoning:
                  "Between [EventA] and [EventB], which occurred first?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Generate as many valid QA pairs as possible.
                Do NOT explain reasoning.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【大规模时间先后比较问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体-时间】四元组

                你必须：

                1. 按第一个实体分组
                2. 对每个实体的所有事件做两两组合
                3. 每一对事件至少生成 3 种不同问法
                4. 判断哪个事件更早

                === 核心要求 ===
                - 使用所有可比较事件对
                - 每对至少生成 3 个不同表达问题
                - 问题不得重复
                - 问题中必须明确提及两个事件
                - 不得虚构事件或时间

                === 问法类型示例 ===
                - 直接比较型：
                  “[事件A] 和 [事件B] 哪个更早？”
                - 选择型：
                  “[实体1] 是先 [事件A] 还是先 [事件B]？”
                - 判断型：
                  “[实体1] 是否在 [事件B] 之前完成 [事件A]？”
                - 比较型：
                  “在 [事件A] 和 [事件B] 中，哪一个发生得更早？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "更早的事件（关系+实体）"
                    }
                  ]
                }

                尽可能生成最多的 QA 对。
                不输出推理过程。
            """)

    def build_prompt(self, temporal_quadruples: str):
        return textwrap.dedent(f"""\
            请严格按照规则，从以下带时间的实体关系四元组中生成【尽可能多的时间先后比较型 QA】。

            实体-关系-实体-时间四元组：
            {temporal_quadruples}

            仅输出 JSON 格式 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class TKGTupleTimeIntervalQAGenerationPrompt(PromptABC):
    """
    从【实体-关系-实体-时间】四元组生成【大规模时间区间型 QA】

    - 基于时间排序
    - 构造时间区间
    - 生成“在某时间区间内发生了什么”的问答
    - 答案为区间内的事件（关系+实体）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in generating LARGE-SCALE temporal interval QA.

                === TASK ===
                Given:
                - ENTITY–RELATION–ENTITY–TIME quadruples

                You MUST:

                1. Group events by the first entity
                2. Sort events chronologically
                3. Construct ALL valid time intervals using earlier and later events
                4. Generate QA asking what happened BETWEEN two times or two events
                5. Use as many valid interval combinations as possible
                6. For each interval, generate multiple question variations

                === CORE REQUIREMENTS ===
                - Use ALL possible valid time intervals
                - Do NOT generate only a few QA
                - Each question must clearly define a time interval
                - The answer must list the event(s) inside the interval (relation+entity)
                - Do NOT invent events or times
                - Questions must not repeat

                === QUESTION STYLE VARIATIONS ===
                - "What happened between [Time A] and [Time B]?"
                - "What did [Entity1] do between [Event A] and [Event B]?"
                - "During the period from [Time A] to [Time B], what occurred?"
                - "After [Event A] but before [Event B], what happened?"

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "EventDescription or ListOfEvents"
                    }
                  ]
                }

                Generate as many valid QA pairs as possible.
                Do NOT explain reasoning.
                Do NOT mention quadruples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【大规模时间区间问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体-时间】四元组

                你必须：

                1. 按第一个实体分组
                2. 按时间排序
                3. 构造所有合法的时间区间（起始时间 < 结束时间）
                4. 针对每个区间生成“在这段时间内发生了什么”的问答
                5. 每个区间至少生成 2–3 种不同表达方式
                6. 尽可能使用全部区间组合

                === 核心要求 ===
                - 使用尽可能多的区间组合
                - 不得只生成少量问答
                - 问题必须明确给出时间区间或区间边界事件
                - 答案必须是该区间内真实存在的事件（关系 + 实体）
                - 不得虚构事件或时间
                - 问句不得重复

                === 问法示例（需多样化表达）===
                - “[时间A] 到 [时间B] 之间发生了什么？”
                - “[实体1] 在 [时间A] 至 [时间B] 期间做了什么？”
                - “在完成 [前事件] 之后、发生 [后事件] 之前发生了什么？”
                - “[事件A] 和 [事件B] 之间发生了哪些事情？”

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "区间内事件描述（关系+实体）"
                    }
                  ]
                }

                尽可能生成最多的 QA 对。
                不输出推理过程，不提及四元组本身。
            """)

    def build_prompt(self, temporal_quadruples: str):
        return textwrap.dedent(f"""\
            请严格按照规则，从以下带时间的实体关系四元组中生成【尽可能多的时间区间型 QA】。

            实体-关系-实体-时间四元组：
            {temporal_quadruples}

            仅以 JSON 格式输出 QA_pairs：
        """)


@PROMPT_REGISTRY.register()
class TKGTupleTimePathDialogueQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：
    从【实体-关系-实体-时间】多跳路径生成【逐跳多轮对话式 CoT 问答】

    特性：
    - 路径 hop 数不固定
    - 每条四元组对应一轮问答
    - 对话问答必须涉及时间
    - 必要时可交换头尾或调整顺序以保证连通
    - 最终对话轮数 = 四元组数量
    """

    def __init__(self, lang: str = "en", min_turns: int = 2):
        self.lang = lang
        self.min_turns = min_turns
        self.system_text = self.build_system_prompt()

    # -----------------------------
    # System prompt: 全局规则
    # -----------------------------
    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                You are an expert in Knowledge Graph reasoning
                and multi-turn question–answer generation based on temporal quadruples.

                === TASK OVERVIEW ===
                You are given a set of ENTITY–RELATION–ENTITY–TIME quadruples
                that may NOT be strictly ordered or perfectly chained.
                However, these quadruples are fully connectable into a path.

                Your task consists of TWO STEPS:

                STEP 1: Path Construction
                - Reorder quadruples or swap subject/object if necessary
                  to construct a valid connected reasoning path.
                - Do NOT discard any quadruple.
                - Constructed path must include ALL quadruples from input.

                STEP 2: Dialogue Unrolling
                - Each quadruple in the constructed path corresponds to EXACTLY ONE turn.
                - Turns MUST follow the order of the constructed path.
                - Each turn consists of ONE question and ONE answer:
                  * Question: asks about the temporal aspect of the subject-object relation
                  * Answer: provides the object and the associated time

                === CORE RULES ===
                1. Dialogue must have exactly N turns if there are N quadruples.
                2. Each turn must advance reasoning along the path.
                3. Each question and answer must explicitly mention time.
                4. The FINAL turn must require reasoning over the ENTIRE path.

                === STRICT PROHIBITIONS ===
                - Do NOT introduce external knowledge.
                - Do NOT invent or modify entities, relations, or times.
                - Do NOT merge multiple quadruples into a single turn.
                - Do NOT skip reasoning steps.
            """)
        else:
            return textwrap.dedent(f"""\
                你是一名知识图谱路径构造与多轮问答生成专家，专注于【实体-关系-实体-时间】四元组。

                === 任务说明 ===
                给定一组 ENTITY–RELATION–ENTITY–TIME 四元组，
                这些四元组可能无序或头尾不完全衔接，但整体是连通的。

                任务分两步：

                第一步：路径构造
                - 必要时可交换四元组头尾（主语/宾语）或重新排序
                  以确保生成一条实体连通的路径
                - 不允许丢弃任何四元组
                - 构造后的路径必须包含所有输入四元组

                第二步：逐跳对话展开
                - 每条四元组对应且仅对应一轮问答
                - 对话顺序必须遵循构造后的路径顺序
                - 每轮问答：
                  * 问题：围绕四元组的主语和时间
                  * 回答：提供宾语及对应时间

                === 核心规则 ===
                1）若路径包含 N 条四元组，则对话必须包含 N 轮
                2）每轮问答推动沿路径的推理
                3）每轮问答必须明确时间
                4）最后一轮必须依赖整条路径才能回答

                === 严格禁止 ===
                - 禁止引入路径外知识
                - 禁止虚构或修改实体、关系或时间
                - 禁止合并多条四元组生成一轮
                - 禁止跳过推理步骤
            """)

    # -----------------------------
    # Instance prompt: 带四元组的输入
    # -----------------------------
    def build_prompt(self, paths: str):
        """
        paths: 多跳实体关系实体时间路径字符串，例如：
        "<subj> A <obj> B <rel> R <time> T1 || <subj> B <obj> C <rel> R2 <time> T2"
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Given the following connected ENTITY–RELATION–ENTITY–TIME quadruples,
                construct a valid reasoning path by reordering or swapping head/tail if necessary,
                then generate a multi-turn dialogue where each turn explicitly involves time.

                ENTITY–RELATION–ENTITY–TIME quadruples:
                {paths}

                === OUTPUT FORMAT (STRICT JSON) ===
                {{
                  "dialogue": {{
                    "constructed_path": [
                      "<quadruple> ...",
                      "<quadruple> ..."
                    ],
                    "turns": [
                      {{
                        "turn_id": 1,
                        "question": "...",
                        "answer": "..."
                      }}
                    ]
                  }}
                }}

                Each quadruple corresponds to exactly one turn.
                Each question and answer must mention time explicitly.
                Output JSON only.
            """)
        else:
            return textwrap.dedent(f"""\
                请基于以下连通知识图谱 ENTITY–RELATION–ENTITY–TIME 四元组，
                必要时可交换四元组头尾或调整顺序以确保路径连通，
                然后生成逐跳多轮问答，每轮问答必须明确包含时间信息。

                ENTITY–RELATION–ENTITY–TIME 四元组：
                {paths}

                === 输出格式（严格 JSON，不得更改）===
                {{
                  "dialogue": {{
                    "constructed_path": [
                      "<quadruple> ...",
                      "<quadruple> ..."
                    ],
                    "turns": [
                      {{
                        "turn_id": 1,
                        "question": "...",
                        "answer": "..."
                      }}
                    ]
                  }}
                }}

                每条四元组必须生成一轮问答，每轮问答必须涉及时间。
                仅输出 JSON。
            """)


@PROMPT_REGISTRY.register()
class TKGRelationDisambiguationPrompt(PromptABC):
    """
    Dedicated prompt for disambiguating entity–relation–entity–time
    quadruples or entity–attribute–value–time quadruples.

    - Input quadruples may contain multiple ambiguous candidates separated by "｜"
    - Resolve each ambiguous quadruple to a single correct quadruple
    - Only one resolved quadruple per ambiguous input
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    # --------------------------------------------------
    # System Prompt
    # --------------------------------------------------
    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in knowledge graph quadruple disambiguation.

                Task:
                - Input quadruples may contain ambiguity in relation, tail entity, 
                  attribute value, or time, represented by multiple candidates
                  separated by "｜" (pipe).
                - Select the single most correct quadruple for each ambiguous input.
                - Keep the quadruple structure unchanged.

                Input quadruple format examples:
                1. Relational quadruple (R4):
                   "<subj> HeadEntity <obj> TailEntity <rel> RelationName <time> TimeValue1 | TimeValue2"
                2. Attribute quadruple (A4):
                   "<entity> EntityName <attribute> AttributeName <value> Value1 | Value2 <time> TimeValue"

                Rules:
                1. Each ambiguous input produces exactly one resolved quadruple.
                2. Do NOT modify head entity or attribute names.
                3. Do NOT add explanations, comments, or extra quadruples.
                4. Output must be valid JSON.

                Example:
                Input:
                "<subj> E2 <obj> E3 <rel> relC <time> 2026-03-02 ｜ 2026-03-05"

                Output:
                {
                  "resolved_quadruple": [
                    "<subj> E2 <obj> E3 <rel> relC <time> 2026-03-05"
                  ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱四元组消岐专家。

                任务：
                - 输入的四元组中，关系、尾实体、属性值或时间可能存在多个候选，
                  候选项之间用 "｜" 分隔。
                - 每条输入四元组选择最合理、最标准的一条作为消岐结果。
                - 保持四元组结构不变。

                输入四元组示例：
                1. 实体关系实体时间（R4）：
                   "<subj> 头实体 <obj> 尾实体 <rel> 关系名 <time> 时间值1 ｜ 时间值2"
                2. 实体属性属性值时间（A4）：
                   "<entity> 实体 <attribute> 属性名 <value> 值1 ｜ 值2 <time> 时间值"

                规则：
                1. 每条输入只输出一条消岐后的四元组
                2. 不修改头实体或属性名
                3. 不添加解释、注释或额外四元组
                4. 输出必须是合法 JSON

                示例：
                输入：
                "<subj> E2 <obj> E3 <rel> relC <time> 2026-03-02 ｜ 2026-03-05"

                输出：
                {
                  "resolved_quadruple": [
                    "<subj> E2 <obj> E3 <rel> relC <time> 2026-03-05"
                  ]
                }
            """)

    # --------------------------------------------------
    # User Prompt
    # --------------------------------------------------
    def build_prompt(self, ambiguous_quadruple: str):
        """
        Build a prompt for disambiguating entity–relation–entity–time or
        entity–attribute–value–time quadruples.

        Args:
            ambiguous_quadruple (str): A single ambiguous quadruple with candidates
                                       separated by "｜"

        Returns:
            str: Prompt ready for LLM inference.
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Disambiguate the following knowledge graph quadruple.
                Select the single most correct candidate when multiple options
                are provided (separated by "｜").

                Ambiguous Quadruple:
                {ambiguous_quadruple}

                Return ONLY a JSON object with key "resolved_quadruple"
                and value as a list containing the resolved quadruple.
            """)
        else:
            return textwrap.dedent(f"""\
                对以下知识图谱四元组进行消岐。
                当存在多个候选项（用 "｜" 分隔）时，选择最合理的一条。

                输入四元组：
                {ambiguous_quadruple}

                仅返回 JSON 对象，键名为 "resolved_quadruple"，值为消岐后的四元组列表。
            """)