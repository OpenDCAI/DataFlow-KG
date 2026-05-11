import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class KGEntityExtractionPrompt(PromptABC):
    """
    阶段一：
    从文本中抽取具体实体
    不输出关系
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in constructing knowledge graphs. Your task is to extract knowledge entities from the given text.

                █ Guidelines:
                - You MUST extract ALL entities in the text. Do NOT return only the most important or a subset of entities.
                - Perform coreference resolution and canonicalize names where applicable.
                - Remove duplicates.
                - DO NOT output vague / non-specific references such as:
                  "these methods", "this idea", "such systems", "our approach", "it", "they", etc.
                - Only output entities that can be clearly and uniquely identified
                  (e.g., specific organizations, people, events, concrete concepts, methods, models, software names).

                █ Output Format:
                Output ONLY a JSON array of strings:
                ["entity1","entity2",...]
            """)
        else:
            return textwrap.dedent("""\
                你是构建知识图谱领域的专家。你的任务是从给定的文本中提取知识实体。

                █ 任务要求:
                - 必须抽取文本中所有实体，不允许只输出最重要或部分实体
                - 进行指代消解并统一命名
                - 去重
                - 严禁输出模糊、不具体的实体，例如：
                  “这些方法”、“这种策略”、“该系统”、“我们的工作”、“它”、“他们” 等
                - 只输出**可以被明确指认**的实体
                 （例如：具体人物、机构、事件、清晰命名的概念、方法、模型名称、软件等）

                █ 输出格式:
                仅输出 JSON 数组：
                ["实体1","实体2",...]
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract all entities from the text.

                █ Text
                ```
                {text}
                ```
            """)
        else:
            return textwrap.dedent(f"""\
                请从以下文本中抽取所有实体。

                █ 文本
                ```
                {text}
                ```
            """)


@PROMPT_REGISTRY.register()
class KGRelationGenerationPrompt(PromptABC):
    """
    专属 Prompt：基于实体列表 + 来源文本 + 已有三元组，
    生成【结构不重复】的高置信度知识图谱三元组

    新增约束：
    - 不允许生成头实体-尾实体与已有三元组重复的新三元组
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph construction expert.

                Your task:
                Generate NEW knowledge graph triples based on:
                1) a given list of entity names
                2) source texts
                3) existing knowledge graph triples
                4) common sense or external world knowledge

                ===============================
                ENTITY NORMALIZATION RULE (HARD)
                ===============================

                - Entity names are SYMBOLS, not natural language phrases
                - You MUST use entity names EXACTLY as they appear in the entity list
                - STRICT STRING MATCH is required

                Forbidden modifications include (but are not limited to):
                - Adding articles (e.g., "the", "a", "an")
                - Changing capitalization
                - Adding prefixes or suffixes
                - Pluralization or singularization
                - Rewriting or paraphrasing entity names

                ===============================
                HARD CONSTRAINTS (MUST FOLLOW)
                ===============================

                1. Existing triple constraint (PAIR-LEVEL NOVELTY):
                - You are given a list of EXISTING triples
                - You MUST NOT generate any new triple whose
                  (subject, object) pair already appears in the existing triples
                - This restriction applies EVEN IF the relation is different
                - If unsure whether a pair already exists, DO NOT generate it

                2. Quality constraint:
                - Precision is more important than recall
                - If a relation is uncertain, do NOT generate it
                - Avoid speculative or controversial facts

                =========================
                OUTPUT FORMAT (STRICT JSON)
                =========================
                Return ONLY JSON:
                {
                  "inferred_triple": [
                    "<subj> subject <obj> object <rel> relation",
                    "<subj> subject <obj> object <rel> relation",
                  ]
                }

            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱构建专家。

                你的任务：
                基于以下信息生成【新的】知识图谱三元组：
                1）实体名称列表
                2）来源文本
                3）已有的知识图谱三元组
                4）你的常识或外部背景知识

                ===============================
                【实体规范化硬约束】
                ===============================

                - 实体名称是【符号】，不是自然语言短语
                - 生成三元组时，实体名称必须与实体列表中的字符串【完全一致】
                - 必须进行【字符串级完全匹配】

                严格禁止以下行为（包括但不限于）：
                - 添加冠词（如 “the / a / an”）
                - 改变大小写
                - 添加前缀或后缀
                - 单复数变化
                - 改写、同义替换实体名称

                ===============================
                【硬性约束（必须遵守）】
                ===============================

                1）已有三元组去重约束（结构级）：
                - 已提供一组三元组作为【已有知识】
                - 禁止生成任何“头实体 + 尾实体”与已有三元组相同的新三元组
                - 即使关系不同，也视为违规
                - 如果不确定是否重复，请不要生成

                2）质量优先约束：
                - 宁缺毋滥，准确性优先
                - 不生成高度猜测性或有争议的事实

                ===============================
                【输出格式（严格遵守以下格式，严格 JSON）】
                ===============================
                三元组的字符串格式必须与输入完全一致。
                {
                  "inferred_triple": [
                    "<subj> subject <obj> object <rel> relation",
                    "<subj> subject <obj> object <rel> relation",
                  ]
                }
            """)

    def build_prompt(
        self,
        existing_triples: str,
        source_texts: str
    ):
        """
        entity_list: 实体名称列表
        source_texts: 实体来源文本
        existing_triples: 已有知识图谱三元组
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate NEW knowledge graph triples following all constraints above.

                Existing Triples (DO NOT repeat subject-object pairs):
                {existing_triples}

                Source Texts:
                {source_texts}

                Output STRICT JSON only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述所有约束，生成【不与已有三元组头尾重复】的新三元组。

                已有三元组（禁止头-尾重复）：
                {existing_triples}

                来源文本：
                {source_texts}

                仅以 JSON 格式输出：
            """)


@PROMPT_REGISTRY.register()
class KGInferredTripleGenerationPrompt(PromptABC):
    """
    专属 Prompt：
    基于【已有一跳三元组】生成【可逻辑推理得到的新三元组】

    核心原则：
    - 受限推理
    - 无新实体
    - 无臆造
    - 宁缺毋滥
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a strict knowledge graph logical inference expert.

                Your task:
                Generate NEW knowledge graph triples that can be logically inferred
                from the GIVEN existing one-hop triples.

                =========================
                CORE INFERENCE CONSTRAINTS
                =========================
                1. Do NOT invent new entities.
                   - Every subject and object in generated triples MUST already appear
                     in the input triples.
                2. Do NOT invent new facts or use external world knowledge.
                3. Each generated triple MUST be inferable from AT LEAST TWO input triples.
                4. Inference must be logically sound and explicitly derivable.
                5. If no valid inference exists, return an empty list.

                =========================
                ALLOWED INFERENCE PATTERNS
                =========================
                - Chain inference:
                  A --r1--> B, B --r2--> C  ⇒  A --r3--> C
                - Role propagation:
                  X member_of Y, Y related_to Z ⇒ X related_to Z
                - Semantic compression:
                  Reduce multi-step relations into a valid direct relation
                  ONLY if logically implied.

                =========================
                STRICTLY FORBIDDEN
                =========================
                - Adding new entities
                - Adding new relations not implied by the inputs
                - Common-sense guessing
                - Temporal or causal assumptions
                - Rephrasing existing triples
                - Duplicating input triples

                =========================
                OUTPUT FORMAT (STRICT JSON)
                =========================
                Return ONLY JSON:
                {
                  "inferred_triple": [
                    "<subj> subject <obj> object <rel> relation",
                    "<subj> subject <obj> object <rel> relation",
                  ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名严格的知识图谱逻辑推理专家。

                你的任务：
                基于【已给定的一跳知识图谱三元组】，生成【能够通过逻辑推理得到的新三元组】。

                =========================
                核心推理约束（必须严格遵守）
                =========================
                1）禁止创造新实体：
                   - 新三元组中的主语与宾语，必须已经在输入三元组中出现过。
                2）禁止使用外部知识或常识补全。
                3）每一个新三元组，必须至少由【两条已有三元组】推理得到。
                4）推理必须清晰、可解释、语义自洽。
                5）若不存在可推理的新三元组，返回空列表。

                =========================
                允许的推理模式
                =========================
                - 链式推理：
                  A → B，B → C ⇒ A → C
                - 角色传播：
                  X 属于 Y，Y 发生在 Z ⇒ X 发生在 Z
                - 关系压缩：
                  多跳关系在逻辑上可合并为单一关系

                =========================
                严格禁止
                =========================
                - 引入新实体
                - 引入输入中不存在的关系
                - 主观臆测
                - 事实验证
                - 重复已有三元组

                =========================
                输出格式（严格 JSON）
                =========================
                {
                  "inferred_triple": [
                    "<subj> subject <obj> object <rel> relation",
                    "<subj> subject <obj> object <rel> relation",
                  ]
                }
            """)

    def build_prompt(self, triples: str):
        """
        triples: 多行或列表形式的三元组字符串
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Generate inferred knowledge graph triples strictly following the rules above.

                Input triples:
                {triples}

                Output inferred triples in JSON only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下一跳三元组中生成可推理的新三元组。

                输入三元组：
                {triples}

                仅以 JSON 格式输出 inferred_triple：
            """)


@PROMPT_REGISTRY.register()
class KGRelationTripleExtractionPrompt(PromptABC):
    """
    阶段二：
    给定 text + entity 列表
    - 从 text 中抽取 结构化 实体-关系-实体 三元组
    - subject / object 必须严格来自给定实体列表
    - 关系必须明确由 text 支持，禁止臆测
    - 以 JSON 输出
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a rigorous Knowledge Graph relation extraction expert.

                === TASK ===
                Given:
                - A text paragraph
                - A list of entities

                Your goal:
                Extract factual, text-supported (subject, relation, object) triples.

                === STRICT RULES ===
                1) Subj and Obj MUST come ONLY from the given entity list. Do NOT modify them (exact original form). Never invent new entities.
                   - Do NOT use variants from the text that are not in the list, using its **normalized entity names** in the list instead.
                2) A triple must be explicitly supported by the text. If unsure → DO NOT output.
                3) Relations must:
                   - be predicate/verb-like (e.g., is_born_in, uses_method)
                   - If a relation is not a complete verb phrase (like 'related_to', 'part_of'), **normalize it to a full predicate**, e.g., 'is_related_to', 'is_part_of'
                   - use underscore _ to connect multi-word relations
                   - come from explicit semantics in the text
                   - NOT hallucinated
                4) Do NOT merge or infer semantic facts beyond the text.
                5) Avoid duplicates and contradictions.
                6) If no valid triples exist → return {"triple": []}


                === OUTPUT FORMAT (MUST FOLLOW EXACTLY) ===
                Return ONLY pure JSON:
                {
                  "triple": [
                    "<subj> {entity} <obj> {entity} <rel> {relation}",
                    "<subj> {entity} <obj> {entity} <rel> {relation}"
                  ]
                }

                === OUTPUT EXAMPLE ===
                {
                  "triple": [
                    "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by",
                    "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by"
                  ]
                }
                No explanation. No extra fields.
            """)
        else:
            return textwrap.dedent("""\
                你是一名严谨的知识图谱关系抽取专家。

                === 任务 ===
                已知：
                - 一段文本 text
                - 一个实体列表

                目标：
                从 text 中抽取 (实体-关系-实体, subject-relation-object) 三元组。

                === 严格规则 ===
                1）subject 与 object 必须严格来自给定实体列表，**保持原名不改动**，禁止创造新实体。
                   - 不允许使用文本中未规范化的变体，使用其列表中的规范化名称作为代替。
                2）三元组必须在文本中有**清晰语义支持**，如果不确定，宁缺毋滥。
                3）关系：
                   - 必须是谓语形式，例如 is_born_in, uses_method
                   - 如果关系不完整（如 related_to、part_of），**规范化为完整谓语**（例如 is_related_to、is_part_of）
                   - 关系中的多词用下划线 _ 连接
                   - 严禁臆造，必须在文本中明确可推导
                4）不做主观推断，不跨句推理，禁止逻辑外延。
                5）去重，避免矛盾三元组。
                6）若无可抽取内容，返回 {"triple": []}

                === 输出格式（严格遵守）===
                仅返回JSON:
                {
                  "triple": [
                    "<subj> {entity} <obj> {entity} <rel> {relation}",
                    "<subj> {entity} <obj> {entity} <rel> {relation}"
                  ]
                }

                === 输出例子 ===
                {
                  "triple": [
                    "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by",
                    "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by"
                  ]
                }             
                不输出任何解释或多余内容。
            """)

    def build_prompt(self, text, entity_list: str):
        """
        input_json example:
        {
          "source":"..../bitter_lesson.pdf",
          "text_path":"..../bitter_lesson.md",
          "raw_chunk":".... text ....",
          "entity":"search, learning self play, value function, ..."
        }
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract knowledge graph triples strictly following the system rules.

                Input text:
                {text}

                Entity list:
                {entity_list}

                Output only pure JSON:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照系统规则，从文本中抽取实体关系三元组。

                输入文本：
                {text}

                实体列表：
                {entity_list}

                仅输出 JSON：
            """)


@PROMPT_REGISTRY.register()
class KGOneHopQAPathGenerationPrompt(PromptABC):
    """
    专属 Prompt：从给定知识图谱三元组中生成【一跳（one-hop）QA】
    适配：人物 / 组织 / 地点 / 抽象概念 / 学术实体
    输出：结构化 QA pairs，可直接用于 LLM 训练
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph question-answer generation expert.

                Your task:
                Generate ONE-HOP question-answer pairs strictly based on the given knowledge graph triples.

                Definition of ONE-HOP QA:
                - Each question MUST be answerable using exactly ONE triple
                - The answer MUST come directly from that triple
                - Do NOT combine information from multiple triples
                - Do NOT introduce any external or implicit knowledge

                Rules:
                - Do NOT modify the triple content
                - Do NOT infer new facts
                - Do NOT merge triples
                - Do NOT explain reasoning
                - Each triple may generate one or more QA pairs
                - Questions should be natural and fluent

                Allowed question types:
                - Subject → Object (e.g., Who did X train?)
                - Object → Subject (inverse question)
                - Yes/No questions (optional)

                Output format (STRICT JSON):
                {
                  "QA_pairs": [{
                    "question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."
                  }]
                }

                Triples:
                {triples}
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱问答对生成专家。

                你的任务：
                严格基于给定的知识图谱三元组，生成【一跳（one-hop）问答对】。

                一跳 QA 定义：
                - 每个问题只能由【一条三元组】直接回答
                - 不允许跨三元组推理
                - 答案必须直接来自该三元组

                规则：
                - 不允许修改三元组内容
                - 不允许引入新事实或隐含知识
                - 不允许合并多条三元组
                - 不允许解释推理过程
                - 每条三元组可生成一个或多个问答对
                - 问题表述应自然通顺

                允许的问题类型：
                - 主语 → 宾语
                - 宾语 → 主语（反向问题）
                - 是/否问题（可选）

                【输出格式（严格 JSON）】：
                {
                  "QA_pairs": [{
                    "question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."
                  }]
                }

                待处理三元组：
                {triples}
            """)

    def build_prompt(self, triples: str):
        """
        triples: 多行或列表形式的三元组字符串
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate one-hop QA pairs strictly following the rules above.

                Triples:
                {triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下三元组中生成一跳问答对。

                三元组：
                {triples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class KGTwoHopPathQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从给定【二跳（two-hop）知识路径】中生成问答对
    适配：人物 / 组织 / 地点 / 时间 / 抽象概念 / 学术实体
    输出：结构化 QA pairs，可直接用于 LLM 多跳推理训练
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a multi-hop knowledge graph question-answer generation expert.

                Your task:
                Generate as many high-quality and non-redundant QUESTION-ANSWER pairs as possible
                that REQUIRE EXACTLY TWO HOPS of reasoning,
                strictly based on the given two-hop knowledge graph paths.

                === DEFINITION OF TWO-HOP PATH ===
                A two-hop path has the form:
                <subj> A <obj> B <rel> R1 || <subj> B <obj> C <rel> R2

                Example:
                <subj> Henry <obj> Maple Leaves <rel> is_member_of || <subj> Maple Leaves <obj> Polar Lights <rel> released

                === CRITICAL REQUIREMENTS (MUST FOLLOW) ===
                1. Each QA MUST be generated from a two-hop path and require BOTH triples in this path to answer.
                   - If the question can be answered using only ONE triple, it is INVALID.
                2. Do NOT generate any one-hop questions.
                3. Do NOT ask about intermediate entities directly unless the question clearly depends on both hops.
                4. Do NOT introduce external knowledge or assumptions.
                5. Do NOT modify any entity or relation names.
                6. Generate as many high-quality and non-redundant QA pairs as possible.
                7. If multiple valid two-hop reasoning questions can be generated from the same path, generate all of them.
                8. Avoid duplicate or near-duplicate questions.
                9. Avoid questions with the same reasoning chain and the same answer unless they ask from clearly different perspectives.

                === ALLOWED QUESTION PATTERNS ===
                - Composition reasoning:
                  Example: What did Henry belong to that released Polar Lights?
                - Two-step relational inference:
                  Example: Which work was released by the group Henry is a member of?
                - Reverse two-hop reasoning:
                  Example: Who is a member of the group that released Polar Lights?
                - Intermediate-bridged reasoning:
                  Example: Which group connects Henry to Polar Lights?
                - Indirect attribute questions:
                  Example: When was the album released by Henry's group?
                - Entity identification through two-hop evidence:
                  Example: Which entity is connected to Polar Lights through membership in Maple Leaves?

                === FORBIDDEN QUESTION TYPES ===
                - Any question answerable from only one triple
                - Simple subject-object queries
                - Direct inverse questions of a single triple
                - Questions that ignore the connection between the two triples
                - Duplicate or near-duplicate paraphrases

                === OUTPUT FORMAT (STRICT JSON, DO NOT CHANGE) ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    },
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名多跳知识图谱问答生成专家。

                你的任务：
                严格基于给定的【二跳（two-hop）知识路径】，
                尽可能多地生成高质量、非重复的问答对，
                且每个问题必须依赖完整的两跳推理才能回答。

                === 二跳路径定义 ===
                二跳路径形式如下：
                <subj> A <obj> B <rel> R1 || <subj> B <obj> C <rel> R2

                例子：
                <subj> Henry <obj> Maple Leaves <rel> is_member_of || <subj> Maple Leaves <obj> Polar Lights <rel> released

                === 核心强制要求（必须遵守）===
                1. 每个 QA 必须从一条两跳路径中生成，并且必须同时使用该路径中的两条三元组才能回答。
                   - 若只用一条三元组即可回答，则该 QA 无效。
                2. 禁止生成任何一跳问题。
                3. 禁止只询问中间节点，除非问题语义明确依赖两跳。
                4. 禁止引入外部知识或隐含假设。
                5. 不允许修改任何实体或关系名称。
                6. 在保证质量和不重复的前提下，尽可能多地生成问答对。
                7. 如果同一条两跳路径可以生成多个有效的两跳推理问题，应尽量全部生成。
                8. 避免生成重复或高度相似的问题。
                9. 如果多个问题具有相同推理链和相同答案，只有在提问角度明显不同的情况下才保留。

                === 允许的问题类型 ===
                - 组合关系推理：
                  例如：Henry 所属的团体发布了什么？
                - 两步关系推理：
                  例如：Henry 所在团体发布的作品是什么？
                - 反向两跳推理：
                  例如：谁属于发布了 Polar Lights 的团体？
                - 中间节点桥接推理：
                  例如：哪个团体连接了 Henry 和 Polar Lights？
                - 间接属性查询：
                  例如：Henry 的团体发布的作品是什么时候发布的？
                - 基于两跳证据的实体识别：
                  例如：哪个实体通过 Maple Leaves 与 Polar Lights 产生联系？

                === 严禁的问题类型 ===
                - 可由单条三元组回答的问题
                - 简单主谓宾查询
                - 单条三元组的反向提问
                - 无视两条三元组连接关系的问题
                - 重复或高度相似的改写问题

                === 输出格式（严格 JSON，不得更改）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    },
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }
            """)

    def build_prompt(self, paths: str):
        """
        paths: 二跳路径字符串，例如：
        "<subj> Henry <obj> Maple Leaves <rel> is_member_of || <subj> Maple Leaves <obj> Polar Lights <rel> released"
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate TWO-HOP question-answer pairs strictly following the rules above.

                Generate as many high-quality and non-redundant two-hop QA pairs as possible.
                Each QA must require BOTH triples in a two-hop path.
                If a path supports multiple valid reasoning directions, generate all of them.
                Do not generate one-hop questions.

                Two-hop paths:
                {paths}

                Output QA_pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下二跳路径中生成多跳问答对。

                请在保证质量和不重复的前提下，尽可能多地生成两跳问答对。
                每个 QA 都必须依赖某条二跳路径中的两条三元组。
                如果同一条路径支持多个有效推理方向，请尽量全部生成。
                禁止生成一跳问题。

                二跳路径：
                {paths}

                仅以 JSON 格式输出 QA_pairs：
            """)



@PROMPT_REGISTRY.register()
class KGRelationTripleSubgraphNumericQAPrompt(PromptABC):
    """
    专属 Prompt：从【实体-关系-实体】三元组生成【数字型 QA】，
    每个 QA 必须依赖至少两个三元组
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph QA generation expert.

                === TASK ===
                Given:
                - A set of ENTITY–RELATION–ENTITY triples
                - Multiple entities and multiple triples exist

                Generate numeric QA pairs such that:

                === CORE REQUIREMENTS ===
                1. The answer MUST be a NUMBER
                2. Each question MUST rely on **at least two triples**
                   (e.g., counts, sums, differences, comparisons across triples)
                3. Only use the given triples; do not introduce external knowledge

                Examples:
                - "How many companies did A acquire?" → numeric answer (requires counting multiple acquisition triples)
                - "What is the total number of employees in the companies B acquired?" → numeric answer (requires multiple triples)
                - "What is the difference in revenue between A and B?" → numeric answer (requires at least two triples)

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do NOT explain reasoning or mention triples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【数字型问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体】三元组
                - 三元组中可能有多个实体和多条三元组

                目标：
                生成问答对，要求：

                === 核心要求 ===
                1. 答案必须是数字
                2. 每个问题必须依赖 **至少两条三元组**
                   （例如计数、总和、差值、比较等）
                3. 严格使用给定三元组，不允许引入外部知识

                示例：
                - "A 一共收购了多少家公司？" → 数字答案（依赖多条收购三元组）
                - "B 收购的公司总员工数是多少？" → 数字答案（依赖多条三元组）
                - "A 与 B 的收入差是多少？" → 数字答案（依赖至少两条三元组）

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                不输出推理过程，不提及三元组本身。
            """)

    def build_prompt(self, relation_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate **numeric QA pairs** strictly following the rules above.

                Each question MUST rely on at least **two triples**.

                ENTITY–RELATION–ENTITY triples:
                {relation_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下实体关系三元组中生成【数字型问答】。

                每个问题必须依赖至少 **两条三元组**。

                实体-关系-实体三元组：
                {relation_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class KGRelationTripleSubgraphSetQAPrompt(PromptABC):
    """
    专属 Prompt：从【实体-关系-实体】三元组生成【集合型 QA】，
    每个 QA 必须依赖至少两个三元组
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph QA generation expert.

                === TASK ===
                Given:
                - A set of ENTITY–RELATION–ENTITY triples
                - Multiple entities and multiple triples exist

                Generate set-based QA pairs such that:

                === CORE REQUIREMENTS ===
                1. The answer MUST be a SET (a collection of entities/concepts, separated by commas)
                2. Each question MUST rely on **at least two triples**
                   (e.g., listing, grouping, union, intersection across triples)
                3. Only use the given triples; do not introduce external knowledge

                Examples:
                - "Which companies did A acquire in 2023 and 2024?" → set answer (requires multiple acquisition triples)
                - "What are all the products launched by B and C?" → set answer (requires multiple product triples)
                - "Which cities have both branch offices of D and E?" → set answer (requires at least two triples)

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    },
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do NOT explain reasoning or mention triples explicitly.
                Ensure the answer is a clear set (comma-separated entities/concepts without extra text).
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【集合型问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体】三元组
                - 三元组中可能有多个实体和多条三元组

                目标：
                生成问答对，要求：

                === 核心要求 ===
                1. 答案必须是集合（多个实体/概念的组合，用逗号分隔）
                2. 每个问题必须依赖 **至少两条三元组**
                   （例如列举、分组、并集、交集等跨多条三元组的逻辑）
                3. 严格使用给定三元组，不允许引入外部知识

                示例：
                - "A 在 2023 年和 2024 年分别收购了哪些公司？" → 集合答案（依赖多条收购三元组）
                - "B 和 C 发布的所有产品有哪些？" → 集合答案（依赖多条产品相关三元组）
                - "哪些城市同时有 D 和 E 的分公司？" → 集合答案（依赖至少两条三元组）

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    },
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                不输出推理过程，不提及三元组本身。
                确保答案为清晰的集合形式（仅用逗号分隔实体/概念，无额外文字）。
            """)

    def build_prompt(self, relation_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate **set-based QA pairs** strictly following the rules above.

                Each question MUST rely on at least **two triples**.

                ENTITY–RELATION–ENTITY triples:
                {relation_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下实体关系三元组中生成【集合型问答】。

                每个问题必须依赖至少 **两条三元组**。

                实体-关系-实体三元组：
                {relation_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class KGRelationTripleSubgraphMultiTripleQAPrompt(PromptABC):
    """
    专属 Prompt：从【实体-关系-实体】三元组生成问答对，
    不限制具体问答类型，但每个 QA 必须依赖至少两个三元组。
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph QA generation expert.

                === TASK ===
                Given:
                - A set of ENTITY–RELATION–ENTITY triples
                - Multiple entities and multiple triples exist

                Generate QA pairs based on the given triples.

                === CORE REQUIREMENTS ===
                1. Each question MUST rely on at least two triples.
                2. The question type is NOT restricted.
                   It may involve listing, comparison, reasoning, path reasoning,
                   multi-hop reasoning, aggregation, relation inference, or other forms.
                3. The answer can be an entity, a set of entities, a concept, a relation,
                   a short phrase, or a concise natural-language answer, depending on the question.
                4. Only use the given triples; do not introduce external knowledge.
                5. Avoid questions that can be answered using only one triple.

                Examples:
                - "Which album released by A earned B?" 
                  This requires one triple about A releasing the album and another triple about the album earning B.
                - "What is the connection between A and C through B?"
                  This requires at least two triples forming a path.
                - "Which entities are related to both A and B?"
                  This requires comparing multiple triples.
                - "Why can X be considered connected to Y?"
                  This requires combining evidence from multiple triples.

                === OUTPUT FORMAT ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    },
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do NOT explain reasoning or mention triples explicitly.
                Output JSON only.
                Ensure each QA pair requires information from at least two triples.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱问答生成专家。

                === 任务 ===
                已知：
                - 一组【实体-关系-实体】三元组
                - 三元组中可能包含多个实体和多条关系

                目标：
                基于给定三元组生成问答对。

                === 核心要求 ===
                1. 每个问题必须依赖至少两条三元组。
                2. 不限制具体问答类型。
                   可以是列举、比较、路径推理、多跳推理、聚合、关系推断、
                   事实归纳、实体关联分析等任意形式。
                3. 答案形式不做限制。
                   答案可以是单个实体、多个实体组成的集合、概念、关系、
                   短语，或简洁的自然语言回答。
                4. 严格使用给定三元组，不允许引入外部知识。
                5. 避免生成只依赖一条三元组即可回答的问题。

                示例：
                - “A 发布的哪个专辑获得了 B？”
                  该问题需要结合“A 发布专辑”和“专辑获得 B”两条三元组。
                - “A 和 C 之间通过 B 存在什么联系？”
                  该问题需要至少两条三元组构成路径。
                - “哪些实体同时与 A 和 B 有关？”
                  该问题需要比较多条三元组。
                - “为什么可以认为 X 与 Y 存在关联？”
                  该问题需要综合多条三元组中的信息。

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    },
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                不输出推理过程，不显式提及“三元组”。
                只输出 JSON。
                确保每个问答对都必须使用至少两条三元组中的信息才能回答。
            """)

    def build_prompt(self, relation_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate QA pairs strictly following the rules above.

                Each question MUST require information from at least two triples.
                Do not generate questions that can be answered by a single triple.

                ENTITY–RELATION–ENTITY triples:
                {relation_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下实体关系三元组中生成问答对。

                每个问题必须至少依赖两条三元组中的信息。
                不要生成只依赖单条三元组即可回答的问题。

                实体-关系-实体三元组：
                {relation_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class KGMultiHopPathDialogueQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：
    从【连通或部分无序的知识图谱路径】生成【逐跳多轮对话式 CoT 问答】

    特性：
    - 路径 hop 数不固定
    - 每条三元组对应一轮问答
    - 必要时可以交换三元组头尾或调整顺序以确保连通
    - 最终对话轮数 = 三元组数量
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
                and path-based multi-turn question–answer generation.

                === TASK OVERVIEW ===
                You are given a set of knowledge graph triples
                that may NOT be strictly ordered or perfectly chained.
                However, these triples are fully connectable.

                Your task consists of TWO STEPS:

                STEP 1: Path Construction
                - Reorder triples or swap subject/object if necessary
                  to construct a valid connected reasoning path.
                - Do NOT discard any triple.
                - Constructed path must include ALL triples from input.

                STEP 2: Dialogue Unrolling
                - Each triple in the constructed path corresponds to EXACTLY ONE turn.
                - Turns MUST follow the order of the constructed path.
                - Each turn consists of ONE question and ONE answer:
                  * Question: asks about the subject of the triple
                  * Answer: the object of the SAME triple

                === CORE RULES ===
                1. If the constructed path contains N triples,
                   the dialogue MUST contain EXACTLY N turns.
                2. Each turn advances the reasoning process along the path.
                3. The FINAL turn must require reasoning over the ENTIRE path.

                === STRICT PROHIBITIONS ===
                - Do NOT introduce external knowledge.
                - Do NOT invent or modify entities or relations.
                - Do NOT merge multiple triples into a single turn.
                - Do NOT skip reasoning steps.
            """)
        else:
            return textwrap.dedent(f"""\
                你是一名知识图谱路径构造与多轮问答生成专家。

                === 任务说明 ===
                给定一组知识图谱三元组，
                这些三元组可能无序或头尾不完全衔接，但整体是连通的。

                任务分两步：

                第一步：路径构造
                - 必要时可交换三元组头尾（主语/宾语）或重新排序
                  以确保生成一条实体连通的路径
                - 不允许丢弃任何三元组
                - 构造后的路径必须包含所有输入三元组

                第二步：逐跳对话展开
                - 每条三元组对应且仅对应一轮问答
                - 对话顺序必须遵循构造后的路径顺序
                - 每轮问答：
                  * 问题：围绕三元组的主语
                  * 回答：对应三元组的宾语

                === 核心规则 ===
                1）若路径包含 N 条三元组，则对话必须包含 N 轮
                2）每轮问答推动沿路径的推理
                3）最后一轮必须依赖整条路径才能回答

                === 严格禁止 ===
                - 禁止引入路径外知识
                - 禁止虚构或修改实体、关系
                - 禁止合并多条三元组生成一轮
                - 禁止跳过推理步骤
            """)

    # -----------------------------
    # Instance prompt: 带三元组的输入
    # -----------------------------
    def build_prompt(self, paths: str):
        """
        paths: 多跳知识图谱路径字符串，例如：
        "<subj> A <obj> B <rel> R1 || <subj> A <obj> C <rel> R2 || <subj> C <obj> D <rel> R3"
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Given the following connected knowledge graph triples,
                construct a valid reasoning path by reordering or swapping
                head/tail if necessary, then generate a multi-turn dialogue.

                Knowledge graph triples:
                {paths}

                === OUTPUT FORMAT (STRICT JSON) ===
                {{
                  "dialogue": {{
                    "constructed_path": [
                      "<triple> ...",
                      "<triple> ..."
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

                Each triple corresponds to exactly one turn.
                Output JSON only.
            """)
        else:
            return textwrap.dedent(f"""\
                请基于以下连通知识图谱三元组，
                必要时可交换三元组头尾或调整顺序以确保路径连通，
                然后生成逐跳多轮问答对话。

                知识图谱三元组：
                {paths}

                === 输出格式（严格 JSON，不得更改）===
                {{
                  "dialogue": {{
                    "constructed_path": [
                      "<triple> ...",
                      "<triple> ..."
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

                每条三元组必须生成一轮问答。
                仅输出 JSON。
            """)


@PROMPT_REGISTRY.register()
class KGTupleTextGenerationPrompt(PromptABC):
    """
    专属 Prompt：
    从一组知识图谱条目（n元组）生成连贯自然语言文本

    - 输入为实体关系三元组、实体属性三元组或更高维度的 n 元组（列表形式）
    - 输出为一段自然语言文本描述
    - 不要求多跳或路径连通
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    # -----------------------------
    # System prompt: 全局规则
    # -----------------------------
    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in Knowledge Graph reasoning and natural language generation.

                === TASK ===
                You are given a set of knowledge graph entries (tuples) in a list.
                Each tuple may contain multiple fields:
                - Standard triple: <subj> Entity <obj> Entity <rel> Relation
                - Quadruple: <subj> Entity <obj> Entity <rel> Relation <attribute> Value
                - Higher-order n-tuples: may include additional attribute-value pairs

                Your goal:
                - Generate a coherent, fluent paragraph that describes all the facts in the tuples.
                - Preserve all entities, relations, attributes, and values exactly as given.
                - Include all fields present in the tuple (support 3, 4, or more elements).
                - Do NOT introduce any external knowledge or assumptions.
                - You may merge facts naturally for readability.

                === CONSTRAINTS ===
                1. Include all tuples in the description.
                2. Maintain factual accuracy.
                3. No hallucination or added entities.
                4. Use natural, coherent sentences.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱推理与自然语言生成专家。

                === 任务说明 ===
                给定一组知识图谱条目（n元组）列表，每条条目可能包含多个字段：
                - 标准三元组：<subj> 实体 <obj> 实体 <rel> 关系
                - 四元组：<subj> 实体 <obj> 实体 <rel> 关系 <属性> 属性值
                - 更高维 n 元组：可能包含多个属性-值对

                你的目标：
                - 将所有条目事实生成一段连贯自然语言文本
                - 保持条目中所有实体、关系、属性和值完全一致
                - 包含每条条目中所有字段（支持3、4或更多元素）
                - 不允许引入额外知识或假设
                - 可以将多条事实自然融合，提高可读性

                === 约束 ===
                1）必须包含所有输入条目的事实
                2）保证事实准确
                3）禁止虚构实体或关系
                4）文本应自然、连贯
            """)

    # -----------------------------
    # Instance prompt: 带 n 元组输入
    # -----------------------------
    def build_prompt(self, tuples: list):
        """
        tuples: 多条知识图谱条目列表，例如：
        [
          "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by",
          "<subj> Henry <obj> Maple Leaves <rel> forms <attribute> Role <value> Captain",
          "<subj> Tesla Model Y <obj> 4680 Battery <rel> WillUse <attribute1> Time <value1> <attribute2> Location <value2>",
        ]
        """
        tuples_str = "\n".join(tuples)  # 每条条目单独一行
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Generate a coherent natural language paragraph
                that describes the following knowledge graph entries (tuples), preserving all fields:

                Tuples:
                {tuples_str}

                Output text only, fluent and factually accurate.
            """)
        else:
            return textwrap.dedent(f"""\
                请将以下知识图谱条目（n元组）生成一段连贯自然语言描述，保留所有字段：

                条目：
                {tuples_str}

                仅输出文本，要求流畅且事实准确。
            """)