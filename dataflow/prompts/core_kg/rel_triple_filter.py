import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class KGRelationTupleValidityPrompt(PromptABC):
    """
    专属Prompt：
    判断知识图谱条目（n元组）是否在概念上合理（有效）

    - 支持三元组、四元组及更多字段
    - 适配抽象概念 / 具体人物 / 专有名词 / AI学术领域等全场景
    - 输出结构化 JSON，可直接解析
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a strict knowledge graph plausibility judge.

                === TASK ===
                Judge whether each given knowledge graph tuple is SEMANTICALLY PLAUSIBLE according to general real-world knowledge and common sense.

                Definition of a PLAUSIBLE tuple (MUST FOLLOW STRICTLY):
                1. Each entity/field can conceptually take the role implied by the tuple
                2. The relation/attribute semantics are logically consistent at the conceptual level
                3. All fields in the tuple (3 or more) must be conceptually coherent
                4. No need to verify factual existence, only conceptual plausibility

                Definition of an IMPLAUSIBLE tuple (MUST FOLLOW STRICTLY):
                1. Conceptual role mismatch
                2. Semantic contradiction
                3. Logical impossibility

                Important rules (ABSOLUTELY NO EXCEPTIONS):
                - Do NOT modify the tuple text
                - Do NOT explain your reasoning
                - Do NOT generate new tuples
                - Judge only conceptual plausibility (ignore factual verification)
                - Include all fields present in each tuple in your judgment
                - For entities with multiple meanings, always pick the meaning that makes the tuple conceptually plausible

                Output format (STRICT):
                Return a pure JSON object with key "valid", listing all conceptually plausible tuples.
                Example:
                {
                    "valid_triple": [
                        "<subj> Tesla Model Y <obj> 4680 battery <rel> Adopt <Time> Starting third quarter of 2025 <Location> European market",
                        "<subj> Tesla Model Y <obj> 4680 battery <rel> Adopt <Time> Starting third quarter of 2025 <Location> European market""
                    ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名严格的知识图谱条目可行性判定专家。

                === 任务说明 ===
                判断每条给定的知识图谱条目（n元组）在概念上是否合理可行。

                【可行条目定义】
                1. 条目中每个实体/字段在该条目中承担的角色概念上合理
                2. 关系或属性语义逻辑自洽
                3. 条目中所有字段（3个及以上）概念上自洽
                4. 不要求事实存在，只需概念合理

                【不可行条目定义】
                1. 概念角色错配
                2. 语义冲突
                3. 逻辑不可能

                【重要规则】
                - 不允许修改条目文本
                - 不允许解释原因
                - 不允许生成新条目
                - 仅判断概念可行性
                - 每条条目中的所有字段必须纳入判断
                - 对多义实体，总是选择使条目概念合理的意义

                【输出格式（严格）】
                返回 JSON 对象，键名为 "valid"，列出所有概念上可行的条目。
                示例：
                {
                    "valid_triple": [
                        "<subj> Tesla Model Y <obj> 4680 battery <rel> Adopt <Time> Starting third quarter of 2025 <Location> European market",
                        "<subj> Tesla Model Y <obj> 4680 battery <rel> Adopt <Time> Starting third quarter of 2025 <Location> European market"
                    ]
                }
            """)

    def build_prompt(self, candidate_tuples: str):
        """
        candidate_tuples: 多条条目字符串，每条条目用逗号或换行隔开，可能是三元组、四元组或更多维
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please filter valid Knowledge Graph tuples strictly following the rules above.
                Input tuples: {candidate_tuples}
                Output valid tuples only, in JSON format with key "valid_triple":
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照规则筛选出概念上可行的知识图谱条目（n元组）。
                输入条目：{candidate_tuples}
                仅输出有效条目，使用 JSON 格式，键名为 "valid_triple"：
            """)



@PROMPT_REGISTRY.register()
class KGEntityValidityPrompt(PromptABC):
    """
    专属Prompt：让LLM判断候选内容是否为【知识图谱有效实体】
    适配：抽象概念/具体人物/专有名词/AI学术领域 全场景
    输出：结构化结果，可直接解析
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a strict but domain-agnostic Knowledge Graph entity validation expert.

                Your ONLY task:
                Given a list of candidate terms, output ONLY those that are VALID KNOWLEDGE GRAPH ENTITIES.

                ====================================================
                WHAT COUNTS AS A VALID KNOWLEDGE GRAPH ENTITY
                ====================================================

                A candidate term is VALID if it satisfies AT LEAST ONE of the following categories:

                ----------------------------------------------------
                CATEGORY A — Named / Proper Entities (HIGHEST PRIORITY)
                ----------------------------------------------------
                Any term that functions as a NAME, TITLE, LABEL, or UNIQUE IDENTIFIER
                MUST be considered a valid KG entity.

                This includes (but is not limited to):
                - Person names (famous or non-famous)
                - Organization names (companies, institutions, teams, bands)
                - Creative works (books, songs, albums, films, artworks)
                - Project / system / product names
                - Historical events
                - Named theories, laws, principles
                - Dates, years, or specific time expressions (e.g., "1997", "March 13, 2019")

                IMPORTANT:
                - Familiarity, popularity, or commonness is IRRELEVANT.
                - If the term is used as a name, it is ALWAYS valid.

                ----------------------------------------------------
                CATEGORY B — Abstract or Conceptual Entities
                ----------------------------------------------------
                Abstract concepts are valid KG entities IF they:
                1. Have independent and complete semantic meaning
                2. Refer to a recognized concept, theory, or phenomenon
                3. Are not merely actions or modifiers

                Examples:
                - deep learning
                - Moore's law
                - natural language processing
                - computer vision
                - The Bitter Lesson

                ----------------------------------------------------
                CATEGORY C — Concrete Objects or Systems
                ----------------------------------------------------
                Entities referring to concrete objects, systems, or observable phenomena
                are valid KG entities.

                ====================================================
                WHAT COUNTS AS INVALID (ONLY IF NONE ABOVE APPLY)
                ====================================================

                A candidate is INVALID ONLY IF ALL of the following are true:
                - It is NOT a named/proper entity
                - It is NOT a recognized concept
                - It is merely one of the following:
                  - A verb or action (e.g., research, apply, develop, include)
                  - A modifier only (e.g., deep, intelligent, related)
                  - A functional word (e.g., the, of, in, on)
                  - A semantic fragment without standalone meaning

                ====================================================
                CRITICAL OVERRIDING RULE
                ====================================================

                If a term APPEARS to be a name, title, label, or identifier,
                you MUST treat it as a VALID entity, even if:
                - You are unfamiliar with it
                - It seems rare, domain-specific, or unconventional

                ====================================================
                OUTPUT FORMAT (STRICT)
                ====================================================

                You must return a pure JSON array and keep the original spelling/case of the input terms.
                Format example:
                ["deep learning", "Moore's law", "search-based approach"]
            """)
        else:
            return textwrap.dedent("""\
                你是一名严格但通用的知识图谱有效实体判定专家，适用于所有领域。

                你的唯一任务：
                从候选短语中筛选出【有效的知识图谱实体】，并仅输出有效实体。

                ====================================================
                什么是有效的知识图谱实体
                ====================================================

                只要候选项满足以下【任意一类】，即视为有效实体：

                ----------------------------------------------------
                A 类 —— 专有名词 / 命名实体（最高优先级）
                ----------------------------------------------------
                任何作为“名称 / 标题 / 标签 / 唯一指代符号”的词语，
                必须判定为有效实体，包括但不限于：

                - 人名（无论是否知名）
                - 组织 / 公司 / 乐队 / 团体名称
                - 作品名（书籍、歌曲、专辑、电影、艺术品）
                - 项目名 / 系统名 / 产品名
                - 历史事件
                - 已命名的理论、定律、原则
                - 具体日期或时间表达（如 1997、2019年3月13日）

                ⚠️ 是否常见、是否熟悉，与判定无关。

                ----------------------------------------------------
                B 类 —— 抽象概念实体
                ----------------------------------------------------
                满足以下条件的抽象概念属于有效实体：
                1. 具备独立完整语义
                2. 表示概念 / 理论 / 现象
                3. 不是动作或修饰成分

                例如：
                - 深度学习
                - 摩尔定律
                - 自然语言处理
                - 计算机视觉
                - 苦涩的教训

                ----------------------------------------------------
                C 类 —— 具体对象或系统
                ----------------------------------------------------
                指向具体对象、系统或可观测现象的实体均为有效实体。

                ====================================================
                无效实体（仅在不满足以上任一类时）
                ====================================================

                只有在以下条件全部满足时，才判为无效实体：
                - 不是专有名词
                - 不是已知概念
                - 且仅属于以下情况之一：
                  - 动词 / 行为（研究、应用、提出）
                  - 修饰词（深的、智能的、相关的）
                  - 虚词（的、在、关于）
                  - 无法独立指代的语义碎片

                ====================================================
                最高覆盖规则（极其重要）
                ====================================================

                只要一个词“看起来像名字、标题或标签”，
                即使你不认识、没见过，也必须保留为有效实体。

                ====================================================
                输出格式（严格）
                ====================================================
                必须返回一个纯净JSON数组，保留输入词汇的原始拼写/大小写。
                格式示例：
                ["深度学习", "摩尔定律", "搜索式方法"]
            """)

    def build_prompt(self, candidate_entities: str):
        """
        candidate_entities: 纯逗号分隔的字符串，如 "Bitter Lesson, Rich Sutton, Moore's law, Kasparov"
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please filter valid Knowledge Graph entities strictly follow the rules above.
                Input entities: {candidate_entities}
                Output valid entities only (same format as input):
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照规则筛选出有效知识图谱实体。
                输入实体：{candidate_entities}
                仅输出有效实体（与输入格式一致）：
            """)
