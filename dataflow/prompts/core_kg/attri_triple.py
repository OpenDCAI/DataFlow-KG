import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class KGAttributeTupleValidationPrompt(PromptABC):
    """
    专属 Prompt：评测并过滤给定属性条目（n元组）的合理性（跨领域通用，无具体例子）

    输入：
        - extracted_attribute_tuples: 已抽取的属性条目
          可能是三元组、四元组或更多维度
          示例三元组：["<entity>", "<attribute>", "<value>"]
          示例四元组：["<entity>", "<attribute>", "<value>", "<time>"]

    输出：
        - 合理属性条目列表
        - 不合理属性条目列表
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a universal expert in knowledge graph attribute evaluation, applicable to all domains (geography, finance, technology, etc.).

                Core Judgment Rules (MUST FOLLOW for all domains):
                1. A valid attribute tuple [entity, attribute, value, ...] must satisfy:
                   - Each value strictly matches the CORE SEMANTIC CATEGORY of its attribute;
                   - Values are objective, factual descriptions of the entity's attributes (not temporary events/scenarios/usage);
                   - All additional fields (time, context, source, etc.) must also be consistent and conceptually coherent.
                2. Semantic mismatch (invalid cases):
                   - Any field value belongs to a semantic category unrelated to its attribute;
                   - Loose association with the entity, but does not describe the attribute itself.

                Important rules:
                - Do NOT remove or modify any field in the tuple
                - Evaluate all fields consistently
                - Only include tuples that are fully conceptually valid

                Output format (STRICT):
                Return a pure JSON object with key "valid".
                Example:
                {
                    "valid_triple": [
                        "<entity> Entity <attribute> Attribute <value> Value",
                        "<entity> Entity <attribute> Attribute <value> Value <time> 2020-01-01",
                        "<entity> Entity <attribute> Attribute <value> Value <context> Example"
                    ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名跨领域的知识图谱属性评估专家，可适配地理、金融、科技等所有领域。

                核心判断规则（所有领域必须遵守）：
                1. 合法的属性条目 [实体, 属性, 值, ...] 需满足：
                   - 每个值严格匹配该属性的「核心语义范畴」；
                   - 值是实体该属性的「客观事实描述」（非临时事件/场景/用途）；
                   - 额外字段（时间、上下文、来源等）也必须概念上合理、自洽。
                2. 语义范畴不匹配（非法情况）：
                   - 任意字段值所属语义范畴与属性不符；
                   - 值仅与实体松散关联，但未描述属性本身。

                重要说明：
                - 不得删除或修改任何字段
                - 所有字段必须一致性评估
                - 仅保留完全概念合理的条目

                输出格式（严格，无任何额外文本）：
                返回 JSON 对象，键名为 "valid"。
                示例：
                {
                    "valid_triple": [
                        "<实体> 实体 <属性> 属性 <值> 值",
                        "<实体> 实体 <属性> 属性 <值> 值 <时间> 2020-01-01",
                        "<实体> 实体 <属性> 属性 <值> 值 <上下文> 示例"
                    ]
                }
            """)

    def build_prompt(
        self,
        extracted_attribute_tuples: str
    ):
        """
        构建通用型评测prompt（适配所有领域，无具体例子，支持三元组及更多元组）

        Args:
            extracted_attribute_tuples: 已抽取的属性条目（列表或字符串）

        Returns:
            str: 可直接发送给 LLM 的 prompt
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Evaluate the reasonableness of the following attribute tuples based on universal semantic rules (applicable to all domains):
                1. Reject tuples where any value does NOT match the core semantic category of its attribute;
                2. Keep only tuples where all values are objective facts of the entity's attributes and all additional fields are conceptually coherent.

                Extracted Attribute Tuples:
                {extracted_attribute_tuples}

                Return ONLY the JSON object as required (no extra explanation, no example tuples).
            """)
        else:
            return textwrap.dedent(f"""\
                请基于通用语义规则评估以下属性条目（n元组）的合理性（适配所有领域）：
                1. 过滤所有「任意值与属性核心语义范畴不匹配」的条目；
                2. 仅保留「所有值是实体该属性客观事实，且额外字段概念合理」的条目。

                已抽取属性条目：
                {extracted_attribute_tuples}

                仅返回要求的JSON对象，无任何额外解释，无示例条目。
            """)


class KGAttributeTripleExtractionPrompt(PromptABC):
    """
    给定 text + entity 列表
    - 为实体抽取【属性-属性值】信息（非实体-实体关系）
    - entity 必须严格来自给定实体列表
    - attribute / value 必须由 text 明确支持
    - 以 JSON 输出
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a rigorous Knowledge Graph attribute extraction expert.

                === TASK ===
                Given:
                - A text paragraph
                - A list of entities

                Your goal:
                Extract intrinsic attributes for the given entities from the text.

                Attributes are NOT relations between entities.
                They describe properties of a single entity, such as:
                - type / category
                - date / time
                - quantity / number
                - description / definition
                - function / purpose
                - characteristic explicitly stated in the text

                === STRICT RULES ===
                1) The entity MUST come ONLY from the given entity list.
                   - Use the exact original entity name from the list.
                   - Do NOT invent or normalize new entity names.
                2) Attributes must be intrinsic properties of the entity itself.
                   - Do NOT express relationships between two entities.
                   - Do NOT encode relations implicitly as attributes.
                3) Attribute values MUST be explicitly supported by the text.
                   - No inference, no background knowledge, no guessing.
                4) Attributes must be concise noun-like or property-like names:
                   - e.g., type, description, founding_year, function
                   - use underscore _ for multi-word attributes
                5) Do NOT merge information across sentences or entities.
                6) Avoid duplicates and contradictions.
                7) If no valid attributes exist → return {"triple": []}

                === OUTPUT FORMAT (MUST FOLLOW EXACTLY) ===
                Return ONLY pure JSON:
                {
                  "triple": [
                    "<entity> {entity} <attribute> {attribute_name} <value> {value}",
                    "<entity> {entity} <attribute> {attribute_name} <value> {value}"
                  ]
                }

                === OUTPUT EXAMPLE ===
                {
                  "triple": [
                    "<entity> DARPA <attribute> type <value> government agency",
                    "<entity> Moore's law <attribute> description <value> computing power doubles every 18 months"
                  ]
                }

                No explanation. No extra fields.
            """)
        else:
            return textwrap.dedent("""\
                你是一名严谨的知识图谱【实体属性】抽取专家。

                === 任务 ===
                已知：
                - 一段文本 text
                - 一个实体列表

                目标：
                从 text 中为给定实体抽取【属性-属性值】信息。

                注意：这里的属性不是实体之间的关系，
                而是描述单一实体本身的内在属性，例如：
                - 类型 / 类别
                - 时间 / 日期
                - 数量 / 数值
                - 描述 / 定义
                - 功能 / 用途
                - 文本中明确陈述的特征

                === 严格规则 ===
                1）实体（entity）必须严格来自给定实体列表，**保持原名不改动**，禁止创造新实体。
                2）属性必须是实体自身的内在属性：
                   - 不允许抽取实体与实体之间的关系
                   - 不允许用属性形式隐式表达关系
                3）属性值必须在文本中有**明确直接的语义支持**，禁止常识补充与主观推断。
                4）属性名应为简洁的名词或属性短语：
                   - 如 type、description、founding_year、function
                   - 多词属性使用下划线 _ 连接
                5）不跨句整合信息，不跨实体推理。
                6）去重，避免矛盾属性。
                7）若无可抽取属性，返回 {"triple": []}

                === 输出格式（严格遵守）===
                仅输出 JSON：
                {
                  "triple": [
                    "<entity> {entity} <attribute> {attribute_name} <value> {value}",
                    "<entity> {entity} <attribute> {attribute_name} <value> {value}"
                  ]
                }

                === 输出例子 ===
                {
                  "triple": [
                    "<entity> DARPA <attribute> type <value> government agency",
                    "<entity> Moore's law <attribute> description <value> computing power doubles every 18 months"
                  ]
                }

                不输出任何解释或多余内容。
            """)

    def build_prompt(self, text, entity_list: str):
        """
        input_json example:
        {
          "source":".../paper.pdf",
          "raw_chunk":".... text ....",
          "entity":"DARPA, Moore's law, HMMs"
        }
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract entity attributes strictly following the system rules.

                Input text:
                {text}

                Entity list:
                {entity_list}

                Output only pure JSON:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照系统规则，从文本中为实体抽取属性信息。

                输入文本：
                {text}

                实体列表：
                {entity_list}

                仅输出 JSON：
            """)


@PROMPT_REGISTRY.register()
class KGAttributeTripleSingleEntityQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从【实体-属性-属性值】三元组中生成【严格单实体】基础 QA

    - 输入：来自同一知识图谱的一组实体属性三元组（包含多个实体）
    - 输出：每一个 QA 仅涉及【一个实体】
    - 不同 QA 可以使用不同实体
    - 仅使用给定三元组，不引入外部知识
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
                - A set of ENTITY–ATTRIBUTE–VALUE triples
                - All triples come from the same knowledge graph
                - The triples may involve MULTIPLE DIFFERENT ENTITIES

                Generate BASIC question–answer pairs such that:

                === CORE REQUIREMENT ===
                Each QA pair MUST involve **EXACTLY ONE ENTITY**.

                Different QA pairs may use different entities,
                but **ONE QA = ONE ENTITY ONLY**.

                === HARD CONSTRAINT (CRITICAL) ===
                ❌ Any question involving TWO OR MORE entities is INVALID.
                ❌ If more than one entity appears in the question,
                   the QA MUST NOT be generated.

                Every QA MUST:
                - Explicitly mention exactly one entity in the QUESTION
                - Ask about the attributes or attribute values of that entity

                === ALLOWED OPERATIONS (STRICTLY LIMITED) ===
                You may ONLY:
                - Query an attribute value of the entity
                - Verify whether the entity has a specific attribute or value
                - List multiple attributes or values of the entity
                - Perform simple filtering over the entity’s attributes

                You may NOT:
                - Compare the entity with any other entity
                - Introduce additional entities
                - Introduce external or implicit knowledge
                - Invent attributes or values
                - Convert attributes into relations

                === ALLOWED QUESTION TYPES (SINGLE-ENTITY ONLY) ===
                1) Attribute value query  
                   (e.g., What is the population of A?)
                2) Attribute existence verification (Yes/No)  
                   (e.g., Does A have attribute X?)
                3) Attribute or value listing  
                   (e.g., What attributes are known for A?)
                4) Attribute-based condition checking  
                   (e.g., Which of A’s attributes satisfy condition X?)

                === OUTPUT FORMAT (STRICT JSON) ===
                {
                  "QA_pairs": [{
                    "question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."
                  }]
                }

                Do NOT explain reasoning.
                Do NOT mention triples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【单实体属性问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值】三元组
                - 这些三元组来自同一知识图谱
                - 三元组中【可能包含多个不同实体】

                目标：
                生成【基础问答对】，满足：

                === 核心要求 ===
                每一个 QA 对【只涉及一个实体】。

                不同 QA 可以使用不同实体，
                但【一个 QA 中严禁出现第二个实体】。

                === 强制约束（非常重要）===
                ❌ 任何涉及【两个或以上实体】的问题都是【非法的】
                ❌ 如果问题中出现不止一个实体，禁止生成该 QA
                ❌ 严禁任何形式的实体对比、实体引用或实体暗示

                每一个 QA 必须：
                - 在【问题中】显式提及【且仅提及一个实体】
                - 围绕该实体已有的属性或属性值进行提问

                === 允许的操作（严格受限）===
                你只可以：
                - 查询实体的某个属性值
                - 判断实体是否具有某个属性或属性值
                - 枚举实体的多个属性或属性值
                - 基于属性值进行简单条件判断

                严禁：
                - 对实体进行比较或排序
                - 引入其他实体
                - 引入外部或常识性知识
                - 使用未出现的属性或属性值
                - 将属性隐式转化为实体关系

                === 允许的问题类型（仅限单实体）===
                1）实体属性值查询  
                2）实体属性是否存在（是 / 否）  
                3）实体属性或属性值枚举  
                4）基于属性条件的单实体判断

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [{
                    "question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."
                  }]
                }

                不输出推理过程，不提及三元组本身。
            """)

    def build_prompt(self, attribute_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate single-entity QA pairs strictly following the rules above.

                Entity-Attribute-Value triples:
                {attribute_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下实体属性三元组中生成【单实体】问答对。

                实体-属性-属性值三元组：
                {attribute_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class KGAttributeTripleMultiEntityBaseQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从【实体-属性-属性值】三元组中生成多实体基础 QA

    - 输入：从同一知识图谱中采样得到的实体属性三元组
    - 采样规则：围绕一个或多个指定实体
    - 输出：涉及【两个或以上实体】的基础问答对
    - 仅使用给定三元组，不引入外部知识
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
                - A set of ENTITY–ATTRIBUTE–VALUE triples
                - All triples are sampled from the same knowledge graph
                - Triples are related to one or more target entities

                Generate BASIC question–answer pairs that involve
                **AT LEAST TWO DISTINCT ENTITIES**.

                === HARD CONSTRAINT (CRITICAL) ===
                ❌ Any question involving ONLY ONE entity is INVALID.
                ❌ If fewer than two entities are mentioned in the question,
                   the QA MUST NOT be generated.

                Every QA MUST:
                - Explicitly mention two or more entities in the QUESTION
                - Compare, verify, or aggregate their attributes

                === ALLOWED OPERATIONS (STRICTLY LIMITED) ===
                You may ONLY:
                - Compare attributes across multiple entities
                - Find shared attributes or shared attribute values
                - Find attribute differences between entities
                - Verify whether multiple entities share a property

                You may NOT:
                - Introduce external or implicit knowledge
                - Use information outside the given triples
                - Invent attributes or values
                - Convert attributes into hidden relations

                === ALLOWED QUESTION TYPES (MULTI-ENTITY ONLY) ===
                1) Attribute intersection
                   (e.g., What attributes do A and B share?)
                2) Attribute difference
                   (e.g., What does A have that B does not?)
                3) Multi-entity verification (Yes/No)
                   (e.g., Do A and B belong to the same category?)
                4) Multi-entity aggregation
                   (e.g., Which cities among A, B, and C have population > X?)

                === GENERATION COVERAGE REQUIREMENT ===
                You MUST generate as many valid QA pairs as possible from the given triples.

                Specifically:
                - Enumerate ALL semantically distinct QA pairs that satisfy the rules above
                - Consider all applicable combinations of entities, attributes, attribute values,
                  and allowed question types
                - Do NOT stop after generating only a few examples
                - Do NOT omit a valid QA simply because other QAs have already been generated
                  from the same triples
                - Do NOT generate duplicate QAs or paraphrases with the same meaning

                If no valid multi-entity QA can be formed, return an empty list.

                === OUTPUT FORMAT (STRICT JSON) ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do NOT explain reasoning.
                Do NOT mention triples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【多实体属性问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值】三元组
                - 这些三元组来自同一知识图谱
                - 三元组围绕若干实体采样得到

                目标：
                生成【基础问答对】，且——

                === 强制约束（非常重要）===
                ❌ 任何只涉及【一个实体】的问题都是【非法的】
                ❌ 如果问题中未明确出现【两个或以上不同实体】，禁止生成该 QA

                每一个 QA 必须：
                - 在【问题中】显式提及 ≥2 个实体
                - 基于这些实体的属性进行对比、验证或聚合

                === 允许的操作（严格受限）===
                你只可以：
                - 计算多个实体的属性或属性值交集
                - 比较多个实体的属性差异
                - 判断多个实体是否共享某一属性或属性值
                - 对多个实体进行属性条件筛选

                严禁：
                - 引入外部或常识性知识
                - 使用未出现的属性或属性值
                - 将属性隐式转化为实体关系
                - 生成任何单实体问答

                === 允许的问题类型（仅限多实体）===
                1）实体属性交集查询  
                2）实体属性差异查询  
                3）多实体是 / 否 验证  
                4）多实体属性条件筛选

                === 生成覆盖要求 ===
                你必须基于给定三元组，尽可能完整地生成所有符合要求的问答对。

                具体要求：
                - 枚举所有满足上述规则、且语义不同的有效 QA
                - 需要尽可能考虑所有可用的实体组合、属性、属性值以及允许的问题类型
                - 不要只生成少量示例后就停止
                - 不要因为某些三元组已经用于生成过其他 QA，就遗漏仍然合法的新 QA
                - 不要生成语义重复的 QA，也不要仅生成同义改写

                如果无法构造任何合法的多实体 QA，则返回空列表。

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

    def build_prompt(self, attribute_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate ALL valid and semantically distinct multi-entity QA pairs
                that can be constructed from the triples below, strictly following the rules above.
                Do not generate only a small number of examples.

                Entity-Attribute-Value triples:
                {attribute_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下实体属性三元组中，
                尽可能完整地生成【所有符合要求且语义不同】的多实体问答对。
                不要只生成少量示例。

                实体-属性-属性值三元组：
                {attribute_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)

@PROMPT_REGISTRY.register()
class KGAttributeTripleMultiAttributeBaseQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从【同一实体的多个 实体-属性-属性值 三元组】中生成多属性基础 QA

    - 输入：具有相同实体的实体属性三元组
    - 输出：必须联合使用【两个或以上三元组】的基础问答对
    - 每个 QA 必须涉及同一实体的【两个或以上属性】
    - 仅使用给定三元组，不引入外部知识
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph multi-attribute QA generation expert.

                === TASK ===
                Given:
                - A set of ENTITY–ATTRIBUTE–VALUE triples
                - All triples describe the SAME entity

                Generate BASIC question–answer pairs that require
                **AT LEAST TWO DISTINCT TRIPLES** to answer.

                === HARD CONSTRAINT (CRITICAL) ===
                ❌ Any QA that can be answered using only ONE triple is INVALID.
                ❌ Any question involving only ONE attribute of the entity is INVALID.

                Every QA MUST:
                - Be about the same entity
                - Jointly use two or more different attributes of that entity
                - Require information from at least two distinct triples to answer

                === ALLOWED OPERATIONS (STRICTLY LIMITED) ===
                You may ONLY:
                - Ask for multiple attribute values of the same entity
                - Verify whether the entity simultaneously satisfies multiple attribute–value conditions
                - Ask for one attribute value under the condition of one or more other known attributes
                - Summarize the entity using multiple explicitly given attributes

                You may NOT:
                - Introduce external or implicit knowledge
                - Use information outside the given triples
                - Invent attributes or values
                - Generate any QA that depends on only one triple
                - Convert attributes into hidden relations

                === ALLOWED QUESTION TYPES (MULTI-ATTRIBUTE ONLY) ===
                1) Multi-attribute retrieval
                   (e.g., What are the nationality and occupation of A?)
                2) Multi-attribute verification (Yes/No)
                   (e.g., Is A a singer and a Canadian?)
                3) Conditional attribute query
                   (e.g., Given that A is a singer, what is A's nationality?)
                4) Multi-attribute summary
                   (e.g., What information is known about A's occupation and nationality?)

                === GENERATION COVERAGE REQUIREMENT ===
                You MUST generate as many valid QA pairs as possible from the given triples.

                Specifically:
                - Enumerate ALL semantically distinct QA pairs that satisfy the rules above
                - Consider all valid combinations of two or more distinct attributes / triples
                - For each valid attribute combination, consider all applicable allowed question types
                - Do NOT stop after generating only a few examples
                - Do NOT omit a valid QA simply because the same triples have already been used
                  in another QA
                - Do NOT generate duplicate QAs or paraphrases with the same meaning

                If no valid multi-attribute QA can be formed, return an empty list.

                === OUTPUT FORMAT (STRICT JSON) ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Do NOT explain reasoning.
                Do NOT mention triples explicitly.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱【单实体多属性问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值】三元组
                - 所有三元组描述的是【同一个实体】

                目标：
                生成【基础问答对】，且每个问答对都必须联合使用
                【两个或以上不同三元组】中的信息。

                === 强制约束（非常重要）===
                ❌ 任何只依赖【一个三元组】即可回答的问题都是【非法的】
                ❌ 任何只涉及该实体【一个属性】的问题都是【非法的】

                每一个 QA 必须：
                - 围绕同一个实体展开
                - 联合使用该实体的【两个或以上不同属性】
                - 回答时必须依赖【两个或以上不同三元组】的信息

                === 允许的操作（严格受限）===
                你只可以：
                - 同时查询该实体的多个属性值
                - 判断该实体是否同时满足多个属性-属性值条件
                - 在已知一个或多个属性信息的条件下，查询该实体的另一个属性
                - 基于多个已给属性，对该实体进行基础信息概括

                严禁：
                - 引入外部或常识性知识
                - 使用未出现的属性或属性值
                - 生成任何只依赖单个三元组即可回答的问题
                - 将属性隐式转化为实体关系

                === 允许的问题类型（仅限多属性）===
                1）多属性联合查询  
                   例如：A 的国籍和职业分别是什么？
                2）多属性是 / 否验证  
                   例如：A 是否既是歌手，又来自加拿大？
                3）条件属性查询  
                   例如：已知 A 的职业是歌手，那么 A 的国籍是什么？
                4）多属性基础概括  
                   例如：关于 A 的职业和国籍，已知哪些信息？

                === 生成覆盖要求 ===
                你必须基于给定三元组，尽可能完整地生成所有符合要求的问答对。

                具体要求：
                - 枚举所有满足上述规则、且语义不同的有效 QA
                - 尽可能考虑所有由【两个或以上不同属性 / 三元组】构成的合法组合
                - 对于每一种合法属性组合，尽可能生成所有适用的问题类型
                - 不要只生成少量示例后就停止
                - 不要因为某些三元组已经用于生成过其他 QA，就遗漏仍然合法的新 QA
                - 不要生成语义重复的 QA，也不要仅生成同义改写

                如果无法构造任何合法的多属性 QA，则返回空列表。

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

    def build_prompt(self, attribute_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate ALL valid and semantically distinct multi-attribute QA pairs
                that can be constructed from the triples below, strictly following the rules above.
                Do not generate only a small number of examples.

                Entity-Attribute-Value triples of the same entity:
                {attribute_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下同一实体的属性三元组中，
                尽可能完整地生成【所有符合要求且语义不同】的多属性问答对。
                不要只生成少量示例。

                同一实体的【实体-属性-属性值】三元组：
                {attribute_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)

@PROMPT_REGISTRY.register()
class KGAttributeTripleMultiEntityNumericQAGenrationPrompt(PromptABC):
    """
    专属 Prompt：从【实体-属性-属性值】三元组生成【数字型多实体】QA

    - 输出必须涉及多个实体
    - 答案必须是数字
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
                - ENTITY–ATTRIBUTE–VALUE triples
                - Multiple entities are present

                Generate QA pairs such that:

                === CORE REQUIREMENT ===
                - Each question MUST involve **TWO OR MORE ENTITIES**
                - The answer MUST be a NUMBER
                - Only use the given triples; do not introduce external knowledge

                Every QA MUST:
                - Ask numeric questions that compare, sum, or compute differences across entities
                  (e.g., "What is the difference in population between A and B?" or
                         "What is the total revenue of A and B?")
                - Explicitly mention all involved entities in the question

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
                你是一名知识图谱【多实体数字型问答】生成专家。

                === 任务 ===
                已知：
                - 一组【实体-属性-属性值】三元组
                - 三元组中包含多个实体

                目标：
                生成问答对，要求：

                === 核心要求 ===
                - 每个问题必须涉及两个或更多实体
                - 答案必须是数字
                - 仅使用给定三元组，不引入外部知识

                每一个 QA 必须：
                - 问题围绕实体的数值型属性，进行比较、求和或计算差值
                  （例如：“A 与 B 的人口差是多少？”，“A 和 B 的总收入是多少？”）
                - 在问题中明确提及所有涉及的实体

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

    def build_prompt(self, attribute_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please generate **multi-entity numeric QA pairs** strictly following the rules above.

                ENTITY–ATTRIBUTE–VALUE triples:
                {attribute_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照上述规则，从以下实体属性三元组中生成【数字型多实体】问答对。

                实体-属性-属性值三元组：
                {attribute_triples}

                仅以 JSON 格式输出 NumQA_pairs：
            """)


@PROMPT_REGISTRY.register()
class KGAttributeTripleMultiEntitySetQAGenerationPrompt(PromptABC):
    """
    专属 Prompt：从【实体-属性-属性值（EAV）】三元组生成【集合型 QA】。
    每个 QA 必须依赖至少两条 EAV 三元组，并通过属性空间运算得到答案。
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in knowledge graph QA generation for
                Entity–Attribute–Value (EAV) structured data.

                === TASK DEFINITION ===
                You are given a set of triples in the form:
                (Entity, Attribute, Value)

                Your task is to generate **set-valued question–answer (QA) pairs**
                based on these EAV triples.

                === CORE CONSTRAINTS ===
                1. The answer MUST be a SET, represented as a JSON array.
                2. Each question MUST rely on **at least two EAV triples**.
                3. Each question MUST satisfy at least ONE of the following:
                   - Compare or aggregate the SAME attribute across MULTIPLE entities
                   - Aggregate MULTIPLE values of the SAME attribute for one entity
                4. Questions that can be answered using a single EAV triple are FORBIDDEN.
                5. Use ONLY the given EAV triples. Do NOT introduce external knowledge.
                6. Do NOT output reasoning steps or mention attributes/triples explicitly.

                === ALLOWED QUESTION TYPES ===
                - Attribute intersection:
                  Shared attribute values across multiple entities
                - Attribute union:
                  All distinct attribute values across multiple entities
                - Attribute difference:
                  Attribute values present for one entity but not another
                - Conditional filtering:
                  Entities or values satisfying multiple attribute constraints
                - Attribute grouping:
                  Grouped or collected values under the same attribute
                - Count-based filtering:
                  Values associated with multiple entities
                - Temporal or numeric slicing:
                  Entities or values filtered by attribute value ranges

                === EXAMPLES (ILLUSTRATIVE ONLY) ===
                - What genres are shared by Artist A and Artist B?
                - What genres are associated with Artist A or Artist B?
                - What attributes does A have that B does not?
                - Which entities have the same founding year?
                - What industries are associated with companies founded before 2010?

                === OUTPUT FORMAT (STRICT, COMPACT JSON) ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": ["value1","value2"]
                    }
                  ]
                }

                CRITICAL SERIALIZATION RULE:
                - Output MUST be compact (minified) JSON.
                - Do NOT pretty-print.
                - The "answer" array MUST appear on a single line.
                - Any line break inside "answer" is a formatting error.
            """)
        else:
            return textwrap.dedent("""\
                你是一名专注于【实体–属性–属性值（EAV）】结构的
                知识图谱问答数据生成专家。

                === 任务定义 ===
                已知一组三元组，形式为：
                （实体，属性，属性值）

                你的任务是基于这些 EAV 三元组，生成【集合型问答（Set-valued QA）】。

                === 核心约束 ===
                1. 答案必须是【集合】，以 JSON 数组形式表示。
                2. 每个问题必须依赖 **至少两条 EAV 三元组**。
                3. 每个问题必须满足以下至少一条：
                   - 比较或聚合【多个实体】在【同一属性】下的属性值
                   - 聚合【同一实体】在【同一属性】下的多个属性值
                4. 严禁生成可由【单一 EAV 三元组】直接回答的问题。
                5. 仅允许使用给定 EAV 三元组，不得引入外部知识。
                6. 不输出推理过程，不显式提及属性名或三元组结构。

                === 允许的问题类型 ===
                - 属性交集型：多个实体在同一属性下的共有属性值
                - 属性并集型：多个实体在同一属性下的全部属性值（去重）
                - 属性差集型：某一实体有而另一实体没有的属性值
                - 条件组合型：同时满足多个属性条件的实体或属性值
                - 属性归纳型：同一属性下的所有取值集合
                - 数量筛选型：与多个实体关联的属性值
                - 数值 / 时间区间筛选型：基于属性值范围的集合查询

                === 示例（仅说明形式）===
                - “A 和 B 共有的音乐风格有哪些？”
                - “A 或 B 涉及的所有音乐风格有哪些？”
                - “A 有但 B 没有的属性值有哪些？”
                - “成立年份相同的公司有哪些？”
                - “2010 年之前成立的公司所属行业有哪些？”

                === 输出格式（严格、紧凑 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": ["属性值1","属性值2"]
                    }
                  ]
                }

                【关键序列化规则】
                - 必须输出紧凑（compact / minified）JSON
                - 禁止 pretty-print 或换行格式
                - "answer" 数组必须位于同一行
                - 如 answer 中出现换行符（\\n），视为格式错误
            """)

    def build_prompt(self, attribute_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Generate **set-valued QA pairs** strictly following the system instructions.

                Each question MUST rely on at least **two EAV triples**.

                Entity–Attribute–Value triples:
                {attribute_triples}

                Output ONLY compact, valid JSON:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照系统指令，从以下【实体–属性–属性值】三元组中生成【集合型问答】。

                每个问题必须依赖至少 **两条 EAV 三元组**。

                实体–属性–属性值三元组：
                {attribute_triples}

                仅输出紧凑、合法的 JSON，不得包含任何其他内容：
            """)


@PROMPT_REGISTRY.register()
class KGAttributeNormalizationPrompt(PromptABC):
    """
    专属 Prompt：属性元组规范化（跨领域通用，无具体例子）
    支持三元组、四元组或更多维度元组
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a universal expert in knowledge graph attribute tuple normalization,
                applicable to all domains (geography, finance, technology, etc.).

                Core Normalization Rules (MUST FOLLOW for all domains):
                1. Format standardization:
                   - Keep the original structure of each tuple EXACTLY, e.g., "<entity> [Entity] <attribute> [Attr] <value> [Val]" 
                     or "<entity> [Entity] <attribute> [Attr] <value> [Val] <time> [Time]" or higher-order tuples;
                   - Remove extra spaces, special characters (except necessary hyphens/underscores), and redundant punctuation in all fields;
                   - Unify case rules: attribute names use lowercase, entity/value/time keep original case but remove random capitalization.
                2. Semantic standardization:
                   - Normalize attribute names to universal domain terminology (e.g., "PE" → "price_to_earnings_ratio", "nationality" instead of "country_of_origin");
                   - Normalize values and times to objective, standardized expressions (e.g., "Aug 12, 2020" → "2020-08-12", "25y" → "25 years", "high" → "above average", "2025上半年" → "2025-H1", "春季" → "Spring", "第一个月" → "M1", "第一个季度" → "Q1", ranges like "2022-01-01|2023-12-31").
                3. Consistency:
                   - Ensure the same attribute/entity/time in different tuples uses the same expression.

                Important:
                - Input may contain tuples of any length (3, 4, 5 or more fields); your task is to normalize all tuples consistently.
                - Do NOT remove or filter any tuples; only normalize content according to rules.

                Output Format (STRICT, NO EXTRA TEXT):
                Return a JSON object with ONLY one key "normalized_triple", whose value is a list of normalized tuples.
                Keep the structure of each tuple unchanged, only modify content to meet normalization rules.
            """)
        else:
            return textwrap.dedent("""\
                你是一名跨领域的知识图谱属性元组规范化专家，可适配地理、金融、科技等所有领域。

                核心规范化规则（所有领域必须遵守）：
                1. 格式标准化：
                   - 严格保留每条元组的原始结构，例如 "<entity> [实体] <attribute> [属性] <value> [值]" 
                     或 "<entity> [实体] <attribute> [属性] <value> [值] <time> [时间]" 或更多维度；
                   - 移除所有字段中的多余空格、特殊字符（必要的连字符/下划线除外）、冗余标点；
                   - 统一大小写规则：属性名全小写，实体/值/时间保留原有大小写但去除随机大写。
                2. 语义标准化：
                   - 将属性名规范化为领域通用术语（如"市盈率"而非"PE值"，"国籍"而非"出生地国家"）；
                   - 将值和时间规范化为客观、标准化表述（如"2020年8月12日"→"2020-08-12"，"25岁"→"25 years"，"高"→"高于平均值"，"2025上半年"→"2025-H1"，"春季"→"Spring"，"第一个月"→"M1"，"第一个季度"→"Q1"，范围形式如"2022-01-01|2023-12-31"）。
                3. 一致性：
                   - 确保不同元组中相同实体/属性/时间的表述一致。

                重要说明：
                - 输入可能包含三元组、四元组或更多维度元组；任务是对所有元组统一规范化。
                - 不删除或过滤任何元组，仅根据规范化规则修改内容。

                输出格式（严格，无任何额外文本）：
                返回仅包含一个键"normalized_triple"的JSON对象，值为规范化后的元组列表。
                保留元组结构不变，仅修改内容以符合规范化规则。
            """)

    def build_prompt(self, extracted_attribute_tuples: str):
        """
        构建通用型规范化 prompt（适配所有领域，支持任意维度元组）
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Normalize the following attribute tuples based on universal semantic and format rules (applicable to all domains):
                1. Standardize format (remove extra spaces/special chars, unify case for attributes);
                2. Standardize semantics (use universal terminology for attributes/values/times);
                3. Ensure consistency for the same entity/attribute/time across tuples.

                Extracted Attribute Tuples:
                {extracted_attribute_tuples}

                Return ONLY the JSON object as required (no extra explanation, no example tuples).
                Keep all input tuples (do NOT remove any), only normalize their content.
            """)
        else:
            return textwrap.dedent(f"""\
                请基于通用语义和格式规则规范化以下属性元组（适配所有领域）：
                1. 格式标准化（移除多余空格/特殊字符，统一属性名大小写）；
                2. 语义标准化（属性/值/时间使用领域通用术语）；
                3. 确保相同实体/属性/时间在不同元组中表述一致。

                已抽取属性元组：
                {extracted_attribute_tuples}

                仅返回要求的JSON对象，无任何额外解释。
                保留所有输入元组（不删除任何元组），仅对内容做规范化处理。
            """)


@PROMPT_REGISTRY.register()
class KGAttributeTripleDisambiguationPrompt(PromptABC):
    """
    Dedicated prompt for disambiguating entity attribute triples.
    Only resolves ambiguous attribute values, ignoring other normalization tasks.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in knowledge graph attribute triple disambiguation.

                Task:
                - Input attribute triples may contain multiple candidate values separated by "|" (pipe).
                - Select the most correct, standardized, or widely accepted value.
                - Only disambiguate values; do NOT modify entity names, attribute names, or structure.

                Input triple format:
                "<entity> EntityName <attribute> AttributeName <value> CandidateValue1 | CandidateValue2 | ..."

                Output rules:
                1. Keep triple structure unchanged.
                2. Only choose a single value for each ambiguous attribute.
                3. Do NOT add extra explanation or commentary.

                Example:
                Input: "<entity> Henry <attribute> nationality <value> Canada | Canadian"
                Output:
                {
                  "resolved_attribute": [
                    "<entity> EntityName <attribute> AttributeName <value> ValueName",
                    "<entity> EntityName <attribute> AttributeName <value> ValueName"
                  ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱属性三元组消歧专家。

                任务：
                - 输入的属性三元组可能在 <value> 中包含多个候选值，用 "|" 分隔。
                - 为每个属性选择最正确、标准或通用的值。
                - 只处理属性值的消歧，不修改实体名、属性名或三元组结构。

                输入三元组格式：
                "<entity> 实体名 <attribute> 属性名 <value> 值1 | 值2 | ..."

                输出规则：
                1. 保持三元组结构不变。
                2. 每个有歧义的属性只保留一个值。
                3. 不添加任何额外说明或评论。

                示例：
                输入："<entity> Henry <attribute> nationality <value> Canada | Canadian"
                输出：
                {
                  "resolved_attribute": [
                    "<entity> EntityName <attribute> AttributeName <value> ValueName",
                    "<entity> EntityName <attribute> AttributeName <value> ValueName"
                  ]
                }
            """)

    def build_prompt(self, ambiguous_attribute_triples: str):
        """
        Build a prompt for disambiguating attribute triples.

        Args:
            ambiguous_attribute_triples (str): Attribute triples with ambiguous values (pipe-separated)

        Returns:
            str: Prompt ready to send to LLM
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Disambiguate the following entity attribute triples.
                Only choose the correct value for attributes with multiple candidates (pipe-separated).

                Ambiguous Attribute Triples:
                {ambiguous_attribute_triples}

                Return ONLY a JSON object with key "resolved_attribute" and value as a list of resolved triples.
            """)
        else:
            return textwrap.dedent(f"""\
                对以下属性三元组进行消歧。
                仅为 <value> 中有多个候选值（用 "|" 分隔）的属性选择正确值。

                输入属性三元组：
                {ambiguous_attribute_triples}

                仅返回 JSON 对象，键名为 "resolved_attribute"，值为消歧后的三元组列表。
            """)
