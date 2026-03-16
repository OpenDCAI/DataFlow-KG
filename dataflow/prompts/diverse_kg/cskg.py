import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class CSKGRelationtripleExtractorPrompt(PromptABC):
    """
    从文本中抽取常识性关系三元组（Entity-Relation-Entity）
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a professional expert in extracting Commonsense Knowledge Graph (CSKG) triples from natural language text.
                Your goal is to extract high-quality commonsense triples that reflect daily human experience, physical rules, or social norms.

                === TASK RULES ===
                1. TRIPLE DEFINITION: Each triple is (Concept1, CommonsenseRelation, Concept2), where:
                   - Concept: A clear, concrete/abstract noun/phrase (e.g., "alarm clock", "concentration", "pet dog" — NO pronouns like "she"/"he"/"it")
                   - CommonsenseRelation: A semantic relation that describes "what", "why", "how" (e.g., "UsedFor", "Causes", "CapableOf", "AtLocation", "HasProperty", "Makes", "Helps")
                2. STRICT CONSTRAINTS:
                   - Preserve the core meaning of entities/relations, but normalize to standard common sense wording (e.g., "jog" → "jogging", "listening" → "ListensTo")
                   - Extract ONLY commonsense facts (not just actions) — focus on "why/what for" instead of just "what happened"
                   - Do NOT use pronouns (she/he/it) — replace with the actual entity (e.g., "Alice" instead of "she")
                   - Do NOT invent new concepts/relations/facts — all triples must be implied by the text
                   - Each triple must describe ONE single commonsense fact (avoid complex phrases like "feeding her pet dog")

                === STANDARD COMMONSENSE RELATIONS (REFERENCE) ===
                - UsedFor: X is used for doing Y (e.g., "alarm clock" UsedFor "waking up")
                - Causes: X causes Y (e.g., "coffee" Causes "alertness")
                - CapableOf: X is capable of doing Y (e.g., "human" CapableOf "jogging")
                - AtLocation: X is usually at location Y (e.g., "indoor plants" AtLocation "house")
                - Makes: X makes Y (e.g., "feeding pet" Makes "pet happy")
                - Helps: X helps Y with Z (e.g., "caffeine" Helps "concentration")
                - HasProperty: X has property Y (e.g., "water" HasProperty "liquid")

                === OUTPUT FORMAT ===
                - Output ONLY a JSON object with key "triple" containing a list of triples.
                - Each triple as a string: "<subj> Concept1 <obj> Concept2 <rel> CommonsenseRelation"
                - Use ONLY the standard commonsense relations listed above (or similar semantic relations).
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的常识知识图谱（CSKG）三元组抽取专家，擅长从自然语言文本中提取符合人类日常经验、物理规则、社会规范的常识三元组。

                === 任务规则 ===
                1. 三元组定义：每条三元组为（概念1，常识关系，概念2），其中：
                   - 概念：清晰的具体/抽象名词/短语（如“闹钟”“注意力”“宠物狗”——禁止使用“她/他/它”等代词）
                   - 常识关系：描述“用途/因果/能力/位置/属性”的语义关系（如“用于”“导致”“能够”“位于”“具有属性”“使”“帮助”）
                2. 严格约束：
                   - 保留实体/关系的核心含义，但归一为标准常识表述（如“慢跑”而非“jog”，“听”而非“listening”）
                   - 仅抽取常识性事实（而非单纯动作）——聚焦“为什么/用来做什么”，而非仅“发生了什么”
                   - 禁止使用代词（她/他/它）——替换为具体实体（如用“Alice”代替“she”）
                   - 禁止虚构概念/关系/事实——所有三元组必须由文本推导得出
                   - 每条三元组仅描述一个常识事实（避免“喂她的宠物狗”这类复杂短语）

                === 标准常识关系参考 ===
                - 用于（UsedFor）：X 被用来做 Y（如“闹钟” 用于 “叫醒”）
                - 导致（Causes）：X 导致 Y（如“咖啡” 导致 “清醒”）
                - 能够（CapableOf）：X 能够做 Y（如“人类” 能够 “慢跑”）
                - 位于（AtLocation）：X 通常在 Y 位置（如“室内植物” 位于 “房子”）
                - 使（Makes）：X 使 Y 处于某种状态（如“喂宠物” 使 “宠物开心”）
                - 帮助（Helps）：X 帮助 Y 达成 Z（如“咖啡因” 帮助 “集中注意力”）
                - 具有属性（HasProperty）：X 具有 Y 属性（如“水” 具有属性 “液态”）

                === 输出格式 ===
                - 仅输出 JSON 对象，键为 "triple"，对应三元组列表
                - 每条三元组格式为字符串："<subj> 概念1 <obj> 概念2 <rel> 常识关系"
                - 仅使用上述标准常识关系（或语义相近的规范关系）
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract Commonsense Knowledge Graph (CSKG) triples from the following text, following the rules above:

                Text:
                {text}

                Output ONLY JSON (no extra explanation):
                {{
                  "triple": [
                    "<subj> Concept1 <obj> Concept2 <rel> CommonsenseRelation",
                    "<subj> Concept1 <obj> Concept2 <rel> CommonsenseRelation"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                按照上述规则，从以下文本中抽取常识知识图谱（CSKG）三元组：

                文本：
                {text}

                仅输出 JSON（无额外解释）：
                {{
                  "triple": [
                    "<subj> 概念1 <obj> 概念2 <rel> 常识关系",
                    "<subj> 概念1 <obj> 概念2 <rel> 常识关系"
                  ]
                }}
            """)


@PROMPT_REGISTRY.register()
class CSKGAttributetripleExtractorPrompt(PromptABC):
    """
    从文本中抽取常识性属性三元组（Entity-Attribute-Value）
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a professional expert in extracting Commonsense Knowledge Graph (CSKG) attribute triples from natural language text.
                Your goal is to extract attribute triples that reflect inherent, daily, and universal commonsense properties (NOT temporary actions or events).

                === TASK RULES ===
                1. TRIPLE DEFINITION: Each triple is (Concept, CommonsenseAttribute, AttributeValue), where:
                   - Concept: A clear, concrete/abstract noun/phrase (e.g., "alarm clock", "water", "pet dog" — NO pronouns like "she"/"he"/"it")
                   - CommonsenseAttribute: An inherent/daily attribute that describes "what X is like" (e.g., "state", "color", "function", "material", "taste", "temperature", "shape")
                   - AttributeValue: The value of the attribute (must be a common sense property, NOT temporary actions like "drank" or "jogged")
                2. STRICT CONSTRAINTS:
                   - Preserve the core meaning of concepts/attributes/values, but normalize to standard commonsense wording (e.g., "cold" instead of "feels cold", "liquid" instead of "is liquid")
                   - Extract ONLY commonsense attributes (inherent properties) — focus on "what X is" instead of "what X did"
                   - Do NOT use pronouns (she/he/it) — replace with the actual entity (e.g., "Alice" instead of "she")
                   - Do NOT invent new concepts/attributes/values — all triples must be implied by the text
                   - Each triple must describe ONE single commonsense attribute (avoid complex phrases like "drank 2 glasses of water")

                === STANDARD COMMONSENSE ATTRIBUTES (REFERENCE) ===
                - Physical attributes: color, material, shape, state (solid/liquid/gas), temperature, texture, taste, smell
                - Functional attributes: function, usage, purpose
                - Characteristic attributes: behavior, trait, common state, capability
                - Spatial attributes: location (usual), size, weight

                === OUTPUT FORMAT ===
                - Output ONLY a JSON object with key "triple" containing a list of E-A-V triples.
                - Each triple as a string: "<enity> EntityName <attribute> AttributeName <value> ValueName"
                - Ensure "EntityName" is a clear concept (no pronouns), "AttributeName" is a standard attribute term, "ValueName" is a commonsense value.
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的常识知识图谱（CSKG）属性三元组抽取专家，擅长从自然语言文本中提取反映实体「固有特征、日常属性、通用特性」的常识属性三元组（非临时动作/事件）。

                === 任务规则 ===
                1. 三元组定义：每条三元组为（概念，常识属性，属性值），其中：
                   - 概念：清晰的具体/抽象名词/短语（如“闹钟”“水”“宠物狗”——禁止使用“她/他/它”等代词）
                   - 常识属性：描述“X是什么样的”的固有/日常属性（如“状态”“颜色”“功能”“材质”“味道”“温度”“形状”）
                   - 属性值：属性对应的取值（必须是常识性特征，而非“喝了”“慢跑”等临时动作）
                2. 严格约束：
                   - 保留概念/属性/值的核心含义，但归一为标准常识表述（如“冷”而非“感觉冷”，“液态”而非“是液态的”）
                   - 仅抽取常识属性（固有特征）——聚焦“X是什么”，而非“X做了什么”
                   - 禁止使用代词（她/他/它）——替换为具体实体（如用“Alice”代替“she”）
                   - 禁止虚构概念/属性/值——所有三元组必须由文本推导得出
                   - 每条三元组仅描述一个常识属性（避免“喝了2杯水”这类复杂短语）

                === 标准常识属性参考 ===
                - 物理属性：颜色、材质、形状、状态（固/液/气）、温度、质地、味道、气味
                - 功能属性：功能、用途、目的
                - 特征属性：行为、特质、常见状态、能力
                - 空间属性：常用位置、大小、重量

                === 输出格式 ===
                - 仅输出 JSON 对象，键为 "triple"，对应 E-A-V 三元组列表
                - 每条三元组以字符串表示："<enity> EntityName <attribute> AttributeName <value> ValueName"
                - 确保“EntityName”为清晰概念（无代词），“AttributeName”为标准属性术语，“ValueName”为常识性取值。
            """)

    def build_prompt(self, text: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Extract Commonsense Knowledge Graph (CSKG) attribute triples from the following text, following the rules above:

                Text:
                {text}

                Output ONLY JSON (no extra explanation):
                {{
                  "triple": [
                    "<enity> EntityName <attribute> AttributeName <value> ValueName",
                    "<enity> EntityName <attribute> AttributeName <value> ValueName"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                按照上述规则，从以下文本中抽取常识知识图谱（CSKG）属性三元组（实体-属性-值）：

                文本：
                {text}

                仅输出 JSON（无额外解释）：
                {{
                  "triple": [
                    "<enity> EntityName <attribute> AttributeName <value> ValueName",
                    "<enity> EntityName <attribute> AttributeName <value> ValueName"
                  ]
                }}
            """)


@PROMPT_REGISTRY.register()
class CSKGConceptGeneralizationPrompt(PromptABC):
    """
    对常识知识图谱（CSKG）三元组中的实体概念进行适度泛化，支持关系三元组和属性三元组两种类型
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a professional expert in Commonsense Knowledge Graph (CSKG) concept generalization.
                Your goal is to generalize the entity/concept in CSKG triples to a higher-level but NOT overly broad concept, while preserving the core meaning of the triple.

                === TASK RULES ===
                1. TRIPLE TYPES TO HANDLE:
                   - Type 1 (Relation Triple): "<subj> Concept1 <obj> Concept2 <rel> CommonsenseRelation"
                   - Type 2 (Attribute Triple): "<enity> EntityName <attribute> AttributeName <value> ValueName"
                2. GENERALIZATION RULES:
                   - For Type 1: Generalize "Concept1" (subj) and "Concept2" (obj) to appropriate higher-level concepts (NOT overly broad)
                     Example: "<subj> Toothpaste <obj> Cleaning <rel> UsedFor" → "<subj> Personal Care Product <obj> Cleaning <rel> UsedFor"
                   - For Type 2: Generalize "EntityName" to appropriate higher-level concepts (NOT overly broad)
                     Example: "<enity> Alarm Clock <attribute> Function <value> Waking Up" → "<enity> Timekeeping Device <attribute> Function <value> Waking Up"
                   - Key Constraint: Do NOT over-generalize (e.g., "Toothpaste" → "Personal Care Product" (good) vs "Toothpaste" → "Object" (bad))
                   - Preserve all other parts of the triple (relation/attribute/value) exactly as original
                   - Keep the core commonsense meaning of the triple unchanged
                3. STRICT CONSTRAINTS:
                   - Only generalize the entity/concept part (subj/obj/EntityName), no modification to relation/attribute/value
                   - Use standard commonsense higher-level concepts (avoid rare or overly technical terms)
                   - Do NOT invent new relations/attributes/values, only modify the entity/concept part

                === ACCEPTABLE GENERALIZATION EXAMPLES ===
                - Toothpaste → Personal Care Product (NOT Object/Thing)
                - Apple → Fruit (NOT Food)
                - Alarm Clock → Timekeeping Device (NOT Electronic Product)
                - Running Shoes → Sports Footwear (NOT Footwear/Goods)
                - Coffee Mug → Drinking Vessel (NOT Vessel/Object)

                === OUTPUT FORMAT ===
                - Output ONLY a JSON object with key "gen_triple" containing a list of generalized triples.
                - Keep the triple format exactly the same as input (Type 1/Type 2 structure unchanged)
                - Each triple as a string in the original format:
                  - Type 1: "<subj> GeneralizedConcept1 <obj> GeneralizedConcept2 <rel> OriginalRelation"
                  - Type 2: "<enity> GeneralizedEntityName <attribute> OriginalAttributeName <value> OriginalValueName"
            """)
        else:
            return textwrap.dedent("""\
                你是一名专业的常识知识图谱（CSKG）概念泛化专家，擅长将三元组中的实体/概念泛化为更高层级但**不过度宽泛**的概念，同时保留三元组的核心语义。

                === 任务规则 ===
                1. 需处理的三元组类型：
                   - 类型1（关系三元组）："<subj> 概念1 <obj> 概念2 <rel> 常识关系"
                   - 类型2（属性三元组）："<enity> EntityName <attribute> AttributeName <value> ValueName"
                2. 泛化规则：
                   - 类型1：仅泛化“概念1”（subj）和“概念2”（obj）为合适的高层概念（禁止过度泛化）
                     示例："<subj> 牙膏 <obj> 清洁 <rel> 用于" → "<subj> 洗护用品 <obj> 清洁 <rel> 用于"
                   - 类型2：仅泛化“EntityName”为合适的高层概念（禁止过度泛化）
                     示例："<enity> 闹钟 <attribute> 功能 <value> 叫醒" → "<enity> 计时设备 <attribute> 功能 <value> 叫醒"
                   - 核心约束：泛化范围不能过大（如“牙膏”→“洗护用品”（合理） vs “牙膏”→“物品”（错误））
                   - 保留三元组其他部分（关系/属性/值）完全不变
                   - 保持三元组的核心常识语义不改变
                3. 严格约束：
                   - 仅泛化实体/概念部分（subj/obj/EntityName），不修改关系/属性/值
                   - 使用标准的常识性高层概念（避免生僻或过于专业的术语）
                   - 不虚构新的关系/属性/值，仅修改实体/概念部分

                === 可接受的泛化示例 ===
                - 牙膏 → 洗护用品（而非“物品/东西”）
                - 苹果 → 水果（而非“食物”）
                - 闹钟 → 计时设备（而非“电子产品”）
                - 跑鞋 → 运动 footwear（而非“鞋类/商品”）
                - 咖啡杯 → 饮水容器（而非“容器/物品”）

                === 输出格式 ===
                - 仅输出 JSON 对象，键为 "gen_triple"，对应泛化后的三元组列表
                - 严格保留输入三元组的格式（类型1/类型2结构不变）
                - 每条三元组以原始格式的字符串表示：
                  - 类型1："<subj> 泛化后的概念1 <obj> 泛化后的概念2 <rel> 原始关系"
                  - 类型2："<enity> 泛化后的EntityName <attribute> 原始AttributeName <value> 原始ValueName"
            """)

    def build_prompt(self, triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Generalize the entity/concept in the following CSKG triples according to the rules above (DO NOT over-generalize):

                CSKG Triples:
                {triples}

                Output ONLY JSON (no extra explanation):
                {{
                  "gen_triple": [
                    "Generalized Triple 1",
                    "Generalized Triple 2"
                  ]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                按照上述规则对以下CSKG三元组中的实体/概念进行泛化（禁止过度泛化）：

                CSKG三元组：
                {triples}

                仅输出 JSON（无额外解释）：
                {{
                  "gen_triple": [
                    "泛化后的三元组1",
                    "泛化后的三元组2"
                  ]
                }}
            """)


@PROMPT_REGISTRY.register()
class CSKGSingleRelationTripleQAPrompt(PromptABC):
    """
    CSKG 专属 Prompt（多条三元组）：
    从【多条 Commonsense 实体-关系-实体】三元组生成 QA

    - 输入包含多条 CSKG 实体关系三元组
    - 每条三元组表达的是典型 / 常见 / 合理的常识性关系
    - 为每条三元组独立生成 QA，QA 体现“通常情况下”的常识语义，而非确定事实
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    # =========================
    # System prompt (CSKG rules)
    # =========================
    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in Commonsense Knowledge Graph (CSKG)
                question–answer generation.

                === BACKGROUND ===
                Commonsense knowledge captures what is:
                - typically true
                - generally observed
                - plausible in everyday situations

                It is not absolute, deterministic, or exception-free.

                === TASK ===
                Given MULTIPLE ENTITY–RELATION–ENTITY triples from a Commonsense Knowledge Graph,
                generate a high-quality QA pair for EACH triple independently.

                === CORE RULES ===
                1. Each QA pair must be based ONLY on ONE corresponding triple (one QA per triple).
                2. The question should reflect a **typical or usual situation**.
                3. The answer should be **commonsense-plausible**, not absolute.
                4. Do NOT introduce external knowledge or additional facts.
                5. Do NOT restate the triple verbatim.
                6. Do NOT aggregate or combine information across different triples.

                === QUESTION GUIDELINES ===
                Questions may ask about:
                - what usually happens
                - what people typically do
                - what is commonly associated
                - what is likely or expected

                === ANSWER GUIDELINES ===
                - Concise
                - Natural language
                - No absolute terms such as "always", "never", "guaranteed"

                === STRICT PROHIBITIONS ===
                - No multi-hop reasoning
                - No aggregation across triples
                - No explanations or reasoning steps
                - No encyclopedic or factual tone
                - Do NOT mix information from different triples in one QA pair

                === OUTPUT FORMAT (STRICT JSON) ===
                {
                  "QA_pairs": [
                    {{
                      "question": "...",
                      "answer": "..."
                    }},
                    {{
                      "question": "...",
                      "answer": "..."
                    }}
                  ]
                }

                Ensure the number of QA pairs matches the number of input triples.
                Output JSON only.
            """)
        else:
            return textwrap.dedent("""\
                你是一名【常识知识图谱（CSKG）】多条三元组问答生成专家。

                === 背景说明 ===
                常识知识描述的是：
                - 通常情况下成立的经验性事实
                - 人们普遍认知的合理行为或结果
                - 并非绝对、无例外的真理

                === 任务 ===
                给定【多条】来自常识知识图谱的
                实体-关系-实体三元组，为每条三元组独立生成一组问答。

                === 核心规则 ===
                1）每组问答只能基于对应的单条三元组（一条三元组对应一组QA）  
                2）问题应体现“通常 / 一般情况下”的常识语义  
                3）回答应合理、简洁，但不得绝对化  
                4）禁止引入任何外部知识或额外事实  
                5）不得直接复述三元组原文  
                6）禁止跨三元组整合信息或联合生成问答

                === 问题设计建议 ===
                可围绕：
                - 常见行为
                - 一般性结果
                - 典型关联
                - 合理预期

                === 严格禁止 ===
                - 多跳推理
                - 跨三元组联合/聚合信息
                - 输出推理过程
                - 百科式或事实断言式表述
                - 不同三元组的信息混合到同一组问答中

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

                确保 QA 对的数量与输入三元组的数量一致。
                仅输出 JSON。
            """)

    # =========================
    # Instance prompt
    # =========================
    def build_prompt(self, relation_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Based on the following multiple commonsense triples,
                generate ONE commonsense QA pair for EACH triple independently, following all rules above.

                Commonsense triples:
                {relation_triples}

                Output the QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请基于以下【多条常识知识图谱三元组】，为每条三元组独立生成一组常识性问答。

                常识实体关系三元组：
                {relation_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)



@PROMPT_REGISTRY.register()
class CSKGSetBasedTripleQAPrompt(PromptABC):
    """
    CSKG 专属 Prompt（三元组集合）：
    从【每个 Commonsense 三元组集合】生成 QA，每个 QA 必须用到集合内至少两个三元组的信息

    - 输入包含多个三元组集合（set_triple），每个集合内是语义关联的多条三元组
    - 每条三元组表达典型/常见/合理的常识性关系
    - 为每个集合生成一组 QA，QA 需整合集合内至少两个三元组的信息，体现“通常情况下”的常识语义
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    # =========================
    # System prompt (CSKG rules)
    # =========================
    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in Commonsense Knowledge Graph (CSKG)
                question–answer generation for triple sets.

                === BACKGROUND ===
                Commonsense knowledge captures what is:
                - typically true
                - generally observed
                - plausible in everyday situations

                It is not absolute, deterministic, or exception-free.

                === TASK ===
                Given MULTIPLE SETS of ENTITY–RELATION–ENTITY triples from a Commonsense Knowledge Graph,
                generate ONE high-quality QA pair for EACH set independently.
                Each QA pair MUST integrate information from AT LEAST TWO triples in the corresponding set.

                === CORE RULES ===
                1. Each QA pair must be based ONLY on the corresponding triple set (one QA per set).
                2. The QA pair MUST use information from AT LEAST TWO triples in the set (do NOT use only one triple).
                3. The question should reflect a **typical or usual situation** and integrate multiple commonsense facts.
                4. The answer should be **commonsense-plausible**, not absolute, and summarize the integrated facts.
                5. Do NOT introduce external knowledge or additional facts outside the set.
                6. Do NOT restate triples verbatim; use natural language to integrate information.
                7. Do NOT mix information from different sets in one QA pair.

                === QUESTION GUIDELINES ===
                Questions may ask about:
                - What are the common effects/uses of [entity]?
                - How does [entity] typically affect [aspect1] and [aspect2]?
                - What are the usual benefits of doing [entity]?
                - What properties/relationships does [entity] commonly have?

                === ANSWER GUIDELINES ===
                - Concise and natural language
                - Integrate at least two commonsense facts from the set
                - No absolute terms such as "always", "never", "guaranteed"
                - Summarize rather than list individual triples

                === STRICT PROHIBITIONS ===
                - Generating QA based on only one triple in a set
                - Aggregating information across different sets
                - Explanations or reasoning steps
                - Encyclopedic or factual tone
                - Listing triples instead of integrating them

                === OUTPUT FORMAT (STRICT JSON) ===
                {
                  "QA_pairs": [
                    {{
                      "question": "...",
                      "answer": "..."
                    }},
                    {{
                      "question": "...",
                      "answer": "..."
                    }}
                  ]
                }

                Ensure the number of QA pairs matches the number of input triple sets.
                Output JSON only.
            """)
        else:
            return textwrap.dedent("""\
                你是一名【常识知识图谱（CSKG）】三元组集合问答生成专家。

                === 背景说明 ===
                常识知识描述的是：
                - 通常情况下成立的经验性事实
                - 人们普遍认知的合理行为或结果
                - 并非绝对、无例外的真理

                === 任务 ===
                给定【多个三元组集合】（每个集合内是语义关联的多条常识三元组），
                为每个集合独立生成一组问答。每组问答必须整合该集合内**至少两个三元组**的信息。

                === 核心规则 ===
                1）每组问答只能基于对应的三元组集合（一个集合对应一组QA）  
                2）问答必须用到该集合内**至少两个三元组**的信息（禁止仅用单条三元组）  
                3）问题应体现“通常 / 一般情况下”的常识语义，且整合多个常识事实  
                4）回答应合理、简洁、不绝对化，且汇总集合内的核心常识信息  
                5）禁止引入任何外部知识或额外事实  
                6）禁止直接复述三元组原文；禁止跨集合混合信息  

                === 问题设计建议 ===
                可围绕：
                - [实体] 通常有哪些常见作用/效果？
                - [实体] 一般会对 [方面1] 和 [方面2] 产生什么影响？
                - 做 [实体] 通常有哪些好处？
                - [实体] 普遍具备哪些属性/关系？

                === 严格禁止 ===
                - 仅基于集合内单条三元组生成问答
                - 跨集合整合信息
                - 输出推理过程
                - 百科式或事实断言式表述
                - 罗列三元组而非整合信息

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

                确保 QA 对的数量与输入三元组集合的数量一致。
                仅输出 JSON。
            """)

    # =========================
    # Instance prompt
    # =========================
    def build_prompt(self, set_triples: list):
        # 将三元组集合格式化为易读的文本（每个集合标注序号，便于LLM识别）
        formatted_sets = []
        for idx, triple_set in enumerate(set_triples, 1):
            set_text = f"Set {idx}:\n" + "\n".join([f"  - {triple}" for triple in triple_set])
            formatted_sets.append(set_text)
        sets_str = "\n\n".join(formatted_sets)

        if self.lang == "en":
            return textwrap.dedent(f"""\
                Based on the following sets of commonsense triples,
                generate ONE commonsense QA pair for EACH set independently, following all rules above.
                Each QA pair MUST use information from AT LEAST TWO triples in the corresponding set.

                Commonsense triple sets:
                {sets_str}

                Output the QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请基于以下【多组常识知识图谱三元组集合】，为每个集合独立生成一组常识性问答。
                每组问答必须用到对应集合内**至少两个三元组**的信息，严格遵循上述所有规则。

                常识三元组集合：
                {sets_str}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class CSKGMultiRelationTripleQAPrompt(PromptABC):
    """
    CSKG 专属 Prompt：
    从【Commonsense 实体-关系-实体】三元组生成 QA

    - 输入为 CSKG 中的实体关系三元组
    - 三元组表达的是【典型行为 / 常见因果 / 一般互动】
    - QA 关注“通常 / 一般 / 在常见情况下”
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    # =========================
    # System prompt (CSKG rules)
    # =========================
    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in Commonsense Knowledge Graph (CSKG) reasoning
                and commonsense question–answer generation.

                === BACKGROUND ===
                Commonsense knowledge represents what is:
                - typically true
                - generally plausible
                - commonly observed in everyday situations

                It is NOT absolute, formal, or exception-free knowledge.

                === TASK ===
                Given a set of ENTITY–RELATION–ENTITY triples from a Commonsense Knowledge Graph,
                generate high-quality QA pairs.

                === CORE PRINCIPLES ===
                1. Questions and answers must reflect **typical or usual situations**
                   (e.g., "usually", "generally", "in most cases").
                2. Avoid absolute, deterministic, or overly precise claims.
                3. Do NOT introduce external knowledge or facts not supported by the triples.
                4. Do NOT restate the triples verbatim; express them naturally.

                === QA CONSTRAINTS ===
                - Questions should focus on:
                  • likely actions
                  • common outcomes
                  • typical interactions
                  • plausible effects
                - Answers should be concise and commonsense-plausible.
                - Each QA must rely on **at least two given triples**
                  (e.g., aggregation, comparison, grouping, or joint reasoning).

                === STRICT PROHIBITIONS ===
                - No external knowledge
                - No logical formalism
                - No explanations or reasoning steps
                - No absolute claims such as "always", "never", "guaranteed"

                === OUTPUT FORMAT (STRICT JSON) ===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                Output JSON only.
            """)
        else:
            return textwrap.dedent("""\
                你是一名【常识知识图谱（CSKG）】问答生成专家。

                === 背景说明 ===
                常识知识描述的是：
                - 通常情况下成立的事实
                - 人们普遍认知的经验性规律
                - 合理但并非绝对正确的知识

                它不是严格的、形式化的、无例外的事实。

                === 任务 ===
                给定一组来自【常识知识图谱】的实体-关系-实体三元组，
                生成高质量的问答对。

                === 核心原则 ===
                1）问答必须体现“通常 / 一般 / 常见情况下”的常识语义  
                2）避免绝对化、确定性、精确数值类表述  
                3）严格基于给定三元组，不引入外部知识  
                4）不要直接复述三元组，用自然语言表达  

                === 问答约束 ===
                - 问题应关注：
                  • 常见行为
                  • 可能结果
                  • 一般性互动
                  • 合理推断
                - 回答应简洁、符合常识
                - 每个问答必须依赖 **至少两条三元组**
                  （如：汇总、对比、分组、联合推理）

                === 严格禁止 ===
                - 引入外部事实
                - 形式化逻辑推导
                - 输出推理过程
                - 使用“必然 / 一定 / 永远”等绝对表述

                === 输出格式（严格 JSON）===
                {
                  "QA_pairs": [
                    {
                      "question": "...",
                      "answer": "..."
                    }
                  ]
                }

                仅输出 JSON。
            """)

    # =========================
    # Instance prompt
    # =========================
    def build_prompt(self, relation_triples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Based on the following commonsense ENTITY–RELATION–ENTITY triples,
                generate commonsense QA pairs following all rules above.

                Each question MUST rely on at least **two triples**.

                Commonsense triples:
                {relation_triples}

                Output QA pairs in JSON format only:
            """)
        else:
            return textwrap.dedent(f"""\
                请基于以下【常识知识图谱】实体-关系-实体三元组，
                按照上述所有规则生成常识性问答对。

                每个问题必须依赖至少 **两条三元组**。

                常识实体关系三元组：
                {relation_triples}

                仅以 JSON 格式输出 QA_pairs：
            """)


@PROMPT_REGISTRY.register()
class CSKGTripleAdaptabilityPrompt(PromptABC):
    """
    Evaluate the adaptability of Commonsense Knowledge Graph triples.

    Adaptability measures whether a triple expresses general commonsense
    knowledge that can be reused across contexts rather than a specific event.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):

        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in evaluating Commonsense Knowledge Graph (CSKG) triples.

                Your task is to evaluate the **adaptability** of each triple.

                === DEFINITION OF ADAPTABILITY ===
                Adaptability measures whether the triple expresses general,
                reusable commonsense knowledge that can apply across different
                contexts, rather than describing a specific event.

                === EVALUATION CRITERIA ===
                Consider the following:

                1. GENERALITY
                   - Does the triple represent general knowledge?
                   - Example: "alarm clock UsedFor waking up"

                2. CONTEXT INDEPENDENCE
                   - Can the triple apply in many situations?
                   - Avoid context-specific statements.

                3. ENTITY SPECIFICITY
                   - Triples containing specific people, dates,
                     or one-time events have LOW adaptability.

                4. COMMONSENSE REUSABILITY
                   - The more reusable across contexts, the higher the score.

                === SCORING ===
                Score range: 0 – 1

                1.0 = highly general commonsense knowledge  
                0.5 = partially general but somewhat context-specific  
                0.0 = very specific event or entity-dependent fact  

                === OUTPUT FORMAT ===
                Return ONLY JSON:

                {
                  "adaptability_scores": [float, float, ...]
                }

                Each score corresponds to the triple with the same ID.
                Do NOT output explanations.
            """)

        else:
            return textwrap.dedent("""\
                你是一名常识知识图谱（CSKG）评估专家。

                你的任务是评估每个三元组的 **适应性（adaptability）**。

                === 适应性定义 ===
                适应性表示该三元组是否表达 **通用的常识知识**，
                而不是特定事件或特定人物的事实。

                === 评价标准 ===

                1. 通用性
                   - 是否描述普遍适用的常识
                   - 例如：“闹钟 用于 叫醒”

                2. 上下文独立性
                   - 是否可以在不同情境中复用

                3. 实体特异性
                   - 包含具体人物、时间、一次性事件的三元组适应性较低

                4. 常识复用性
                   - 越容易跨场景复用，分数越高

                === 评分范围 ===
                0 – 1

                1.0 = 高度通用的常识  
                0.5 = 部分通用但有一定上下文依赖  
                0.0 = 非常具体的事件或实体事实  

                === 输出格式 ===
                仅输出 JSON：

                {
                  "adaptability_scores": [float, float, ...]
                }

                每个分数对应输入的一个三元组。
                不要输出解释。
            """)

    def build_prompt(self, triples: list):

        triple_block = ""
        for idx, t in enumerate(triples):
            triple_block += f"ID {idx}: {t}\n"

        if self.lang == "en":
            return textwrap.dedent(f"""\
                Evaluate the **adaptability** of the following CSKG triples.

                --- Triples ---
                {triple_block}

                Return ONLY JSON:
                {{
                  "adaptability_scores": [float, float, ...]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                请评估以下常识三元组的 **适应性（adaptability）**。

                --- 三元组 ---
                {triple_block}

                仅返回 JSON：
                {{
                  "adaptability_scores": [float, float, ...]
                }}
            """)



@PROMPT_REGISTRY.register()
class CSKGTripleRationalePrompt(PromptABC):
    """
    Evaluate the rationale (logical plausibility) of Commonsense Knowledge Graph triples.

    Rationale measures whether the triple represents a logically valid and
    plausible commonsense fact.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):

        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in evaluating Commonsense Knowledge Graph (CSKG) triples.

                Your task is to evaluate the **rationale** of each triple.

                === DEFINITION OF RATIONALE ===
                Rationale measures whether the triple represents a logically
                plausible commonsense fact that makes sense in the real world.

                === EVALUATION CRITERIA ===
                Consider the following:

                1. LOGICAL PLAUSIBILITY
                   - Does the triple make logical sense?

                2. COMMONSENSE VALIDITY
                   - Is the relation between the two concepts consistent with real-world knowledge?

                3. RELATION APPROPRIATENESS
                   - Is the relation correctly describing the connection between the entities?

                4. CONTRADICTION CHECK
                   - Detect impossible or contradictory facts.

                === SCORING ===
                Score range: 0 – 1

                1.0 = highly plausible commonsense fact  
                0.5 = partially plausible or uncertain  
                0.0 = illogical or impossible statement  

                === OUTPUT FORMAT ===
                Return ONLY JSON:

                {
                  "rationale_scores": [float, float, ...]
                }

                Each score corresponds to the triple with the same ID.
                Do NOT output explanations.
            """)

        else:
            return textwrap.dedent("""\
                你是一名常识知识图谱（CSKG）评估专家。

                你的任务是评估每个三元组的 **合理性（rationale）**。

                === 合理性定义 ===
                合理性表示该三元组是否描述了一个符合现实世界常识、
                逻辑上成立的事实。

                === 评价标准 ===

                1. 逻辑合理性
                   - 三元组是否在逻辑上成立

                2. 常识有效性
                   - 实体之间的关系是否符合现实世界常识

                3. 关系匹配度
                   - 关系是否正确描述两个实体之间的联系

                4. 矛盾检测
                   - 是否存在不可能或明显错误的事实

                === 评分范围 ===
                0 – 1

                1.0 = 完全合理的常识事实  
                0.5 = 部分合理或不确定  
                0.0 = 不合理或不可能的事实  

                === 输出格式 ===
                仅输出 JSON：

                {
                  "rationale_scores": [float, float, ...]
                }

                每个分数对应输入的一个三元组。
                不要输出解释。
            """)

    def build_prompt(self, triples: list):

        triple_block = ""
        for idx, t in enumerate(triples):
            triple_block += f"ID {idx}: {t}\n"

        if self.lang == "en":
            return textwrap.dedent(f"""\
                Evaluate the **rationale (logical plausibility)** of the following CSKG triples.

                --- Triples ---
                {triple_block}

                Return ONLY JSON:
                {{
                  "rationale_scores": [float, float, ...]
                }}
            """)
        else:
            return textwrap.dedent(f"""\
                请评估以下常识三元组的 **合理性（rationale）**。

                --- 三元组 ---
                {triple_block}

                仅返回 JSON：
                {{
                  "rationale_scores": [float, float, ...]
                }}
            """)