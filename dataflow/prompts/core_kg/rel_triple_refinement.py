import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class KGEntityTypeClassificationPrompt(PromptABC):
    """
    专属 Prompt：对一组实体字符串列表进行类型判断

    输入：
        - entity_lists: List[str]
          每个元素是以逗号分隔的实体名
          e.g. ["Henry, Maria Rodriguez, Canada", "Maple Leaves, Lucy, Paris"]

    输出：
        - 与输入列表顺序对应的实体类型字符串列表
          e.g. ["PERSON, PERSON, LOCATION", "ORGANIZATION, PERSON, LOCATION"]
    """

    ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "TECHNOLOGY", "EVENT", "OTHER"]

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                You are a knowledge graph entity typing expert.

                Task:
                - Given a list of entity names, classify each entity into one of the following categories:
                  {', '.join(self.ENTITY_TYPES)}

                Rules:
                1. PERSON: Individual human beings
                2. ORGANIZATION: Companies, institutions, government bodies
                3. LOCATION: Geographic places, facilities
                4. CONCEPT: Abstract ideas, theories, methodologies
                5. TECHNOLOGY: Software, hardware, technical systems
                6. EVENT: Specific occurrences, competitions, projects
                7. OTHER: Entities that do not fit the above categories

                Instructions:
                - Classify each entity in order.
                - For multiple entity lists, provide a list of strings with the same length.
                - Output: List[str], each element is a comma-separated entity type string corresponding to input entities.
                - Do NOT include explanations or extra text.

                Example:
                Input: ["Henry, Canada, Maple Leaves", "Lucy, University Toronto"]
                Output: ["PERSON, LOCATION, ORGANIZATION", "PERSON, ORGANIZATION"]
            """)
        else:
            return textwrap.dedent(f"""\
                你是一名知识图谱实体类型分类专家。

                任务：
                - 给定一个实体字符串列表，对每个实体进行类型判断：
                  {', '.join(self.ENTITY_TYPES)}

                类型说明：
                1. PERSON: 个人、人类个体
                2. ORGANIZATION: 公司、机构、政府组织
                3. LOCATION: 地理位置、设施
                4. CONCEPT: 抽象概念、理论、方法论
                5. TECHNOLOGY: 软件、硬件、技术系统
                6. EVENT: 特定事件、比赛、项目
                7. OTHER: 不属于上述类别的实体

                输出要求：
                - 按输入顺序逐个分类
                - 输出 List[str]，每个元素是对应实体类型的字符串（逗号分隔）
                - 不要输出解释或其他文本

                示例：
                输入: ["Henry, Canada, Maple Leaves", "Lucy, University Toronto"]
                输出: ["PERSON, LOCATION, ORGANIZATION", "PERSON, ORGANIZATION"]
            """)

    def build_prompt(self, entity_lists: list):
        """
        构建实体类型判断 prompt

        Args:
            entity_lists: List[str], 每个元素是以逗号分隔的实体名
        
        Returns:
            str: 可直接发送给 LLM 的 prompt
        """
        lists_str = json.dumps(entity_lists)
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Classify the following entity lists according to the rules above.

                Entity Lists:
                {lists_str}

                Output a list of strings, each element containing comma-separated entity types corresponding to input entity strings.
            """)
        else:
            return textwrap.dedent(f"""\
                请根据上述规则，对以下实体字符串列表进行类型分类：

                实体列表：
                {lists_str}

                输出 List[str]，每个元素是对应实体类型的字符串（逗号分隔）。
            """)


class KGEntityDisambiguationPrompt(PromptABC):
    """
    阶段二（实体消歧）：
    给定实体列表 + 对应文本
    - 为每个实体在文本中找到最准确的指向（消歧）
    - 严格根据文本，不允许凭常识或外部知识
    - 输出实体列表格式，顺序对应输入
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a rigorous Knowledge Graph entity disambiguation expert.

                === TASK ===
                Given:
                - A list of entity mentions
                - Corresponding text contexts for each entity

                Your goal:
                Disambiguate each entity mention strictly based on the text.

                === STRICT RULES ===
                1) Use ONLY information explicitly present in the text.
                   - No external knowledge or guessing.
                2) Each entity mention must be resolved independently.
                3) Provide a mapping of entity mention → disambiguated entity.
                4) If no disambiguation can be made → return the original entity name.
                5) Output must be a list of entities, in the same order as input.

                === OUTPUT FORMAT (MUST FOLLOW EXACTLY) ===
                Return ONLY a string with the same format as input entity list:
                e.g.,
                Apple Inc., fruit, Mercury planet, Mercury metal, Python language, Python snake
            """)
        else:
            return textwrap.dedent("""\
                你是一名严谨的知识图谱【实体消歧】专家。

                === 任务 ===
                已知：
                - 一个实体提及列表
                - 每个实体对应的文本上下文

                目标：
                根据文本严格为每个实体提及进行消歧。

                === 严格规则 ===
                1）仅使用文本中明确出现的信息，禁止外部知识或推测。
                2）每个实体独立消歧。
                3）如果无法确定消歧结果 → 返回原实体的名称。
                4）输出必须保持与输入实体列表顺序一致。
                5）输出格式与输入相同，即逗号分隔的实体列表。

                === 输出例子 ===
                输入：
                Apple, Mercury, Python

                输出：
                Apple Inc., Mercury planet, Python language
            """)

    def build_prompt(self, entity_list, text_list):
        """
        entity_list: str, e.g.
        'Henry, Maria Rodriguez, Canada, Maple Leaves, Lucy, University Toronto, Polar Lights, August 12, 2020, Clean Earth, Paris, Berlin, Rome'
        text_list: str, corresponding context for each entity, semicolon or other separator

        Returns formatted prompt
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Disambiguate entities strictly following the system rules.

                Entity mentions:
                {entity_list}

                Corresponding text contexts:
                {text_list}

                Output ONLY a comma-separated entity list, in the same order as input:
            """)
        else:
            return textwrap.dedent(f"""\
                请严格按照系统规则，根据文本为实体提及进行消歧。

                实体提及：
                {entity_list}

                对应文本上下文：
                {text_list}

                仅输出逗号分隔的实体列表，顺序与输入一致：
            """)


@PROMPT_REGISTRY.register()
class KGEntityNormalizationPrompt(PromptABC):
    """
    阶段二：
    对多个 chunk 抽取的实体进行
    - 实体同义聚类（多对一）
    - 标准化命名
    - 输出 JSON
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a strict and professional expert in knowledge graph entity canonicalization and deduplication, with zero tolerance for incorrect merging.
                Your judgment is based ONLY on literal semantic consistency and strict substring inclusion rules, no subjective inference.

                Task:
                - Input: a JSON list of entity names, e.g., ["entity1", "entity2", ...]
                - Some names may refer to the same real-world entity with different surface expressions, need multi-to-one canonicalization.

                === STRICT NON-NEGOTIABLE RULES (MUST FOLLOW ALL, IN ORDER OF PRIORITY) ===
                1) Only normalize entities that have ≥2 semantically identical variants (multi-to-one). Single entity with no variants: NO canonicalization, NO inclusion in output.
                2) [HIGHEST PRIORITY] If a short entity name is a complete substring of a longer entity name AND they refer to the same core concept: 
                   The short name = MUST BE the canonical name, the longer name = MUST BE its variant.
                3) Ignore case differences (e.g., Deep Learning = deep learning), plural forms (e.g., method = methods), minor typos (e.g., learnig = learning).
                4) [CRITICAL] ABSOLUTELY DO NOT merge entities that are semantically distinct, even if they appear in the same context or have similar words.
                   For example: "machine learning" and "deep learning" are DIFFERENT, never merge; "AI" and "ML" are DIFFERENT, never merge.
                5) All variants of a canonical name MUST include ALL original entities that are unified into this canonical name, including the canonical name itself.
                6) The canonical name is the most concise, core, standard expression of the entity.

                === OUTPUT FORMAT (JSON ONLY, NO EXTRA TEXT, NO EXPLANATION, NO COMMENT) ===
                Return a pure JSON object only, keys = canonical names, values = non-empty lists of all original entities unified into that canonical name:
                {{
                  "Canonical Name 1": ["variant_a", "variant_b"],
                  "Canonical Name 2": ["variant_c", "variant_d"]
                }}
            """)
        else:
            return textwrap.dedent("""\
                你是一名严谨专业的知识图谱实体规范化与去重专家，对错误合并实体零容忍。
                仅基于字面语义一致性和严格的子串包含规则做判断，不做任何主观推断。

                任务说明：
                - 输入：一个 JSON 列表，每个元素为独立实体，例如：["实体1", "实体2", ...]
                - 部分实体名称指向同一个真实世界实体但表述不同，需要进行多对一的标准化归一。

                === 严格不可协商的执行规则（全部遵守，按优先级排序）===
                1）仅对拥有≥2个语义完全相同变体的实体做归一化。无变体的单一实体：不做归一，不写入输出结果。
                2）【最高优先级】如果一个短实体名称是另一个长实体名称的完整子串，且二者指向同一个核心概念：
                   短名称 必须作为 标准化名称，长名称 必须作为该标准名的变体。
                3）忽略大小写差异（如 Deep Learning = deep learning）、单复数形式（如 method = methods）、轻微拼写错误（如 learnig = learning）。
                4）【重中之重】绝对不要合并语义不同的实体，即使它们出现在同一上下文或有相似词汇。
                   例如："机器学习" 和 "深度学习" 是不同实体，绝不合并；"AI"和"ML"是不同实体，绝不合并。
                5）标准化名称的所有变体列表必须包含被统一到该名称的所有原始实体，包括标准化名称本身。
                6）标准化名称是实体最简洁、最核心、最标准的表述形式。

                === 输出格式（仅输出JSON，无任何多余文本、无解释、无注释）===
                只返回纯净的JSON对象，键=标准化名称，值=非空的所有原始实体列表：
                {{
                  "标准化名称1": ["实体1", "实体2"],
                  "标准化名称2": ["实体3", "实体4"]
                }}
            """)

    def build_prompt(self, entity_list: str):
        """
        entity_list must be a JSON-formatted list, e.g.,
        '["deep learning", "deep learning methods", "AI"]'
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Process the following JSON list of entity names for knowledge graph canonicalization.
                STRICTLY follow all the priority rules from system prompt, no subjective change.

                Key Reminders:
                1) Short core entity name → canonical name, longer name with it as substring → variant.
                2) Include all original entities unified into a canonical name in its variant list, including the canonical name itself.
                3) Distinct entities with similar words → NEVER merge, even if they appear in same context.
                4) Only output pure JSON object, no other text, no explanation.

                Entity list:
                {entity_list}

                Pure JSON Output Only:
            """)
        else:
            return textwrap.dedent(f"""\
                请对下述实体名称列表执行知识图谱实体标准化归一处理，严格遵守系统提示的所有优先级规则，不做任何主观调整。

                重点提醒：
                1）短的核心实体名作为标准化名称，包含该名称的长实体名作为变体。
                2）变体列表必须包含被统一到该标准化名称的所有原始实体，包括标准化名称本身。
                3）词汇相似但语义不同的实体，绝对不要合并。
                4）仅输出纯净的JSON对象，无任何多余内容。

                实体列表:
                {entity_list}

                仅输出JSON：
            """)


@PROMPT_REGISTRY.register()
class KGRelationNormalizationPrompt(PromptABC):
    """
    Knowledge Graph Relation Direction Normalization Prompt
    Supports triples, quadruples, or higher-order tuples.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph relation direction normalizer.

                Task:
                - Normalize relations into a canonical "is_*" form.
                - Only swap subject and object for agent-based action relations.
                - Input tuples may contain 3 or more elements (e.g., subject, object, relation, time, context).
                - Only the relation and subject/object order should be changed if needed; all other fields remain untouched.

                Relation handling rules:
                1. Agent-based action relations (REQUIRE swapping subject/object):
                   Examples:
                     "develops" -> "is_developed_by"
                     "creates" -> "is_created_by"
                     "sponsors" -> "is_sponsored_by"
                     "produces" -> "is_produced_by"
                2. State, location, or attribute relations (DO NOT swap subject/object):
                   Examples:
                     "performed_in" -> "is_performed_in"
                     "located_in"   -> "is_located_in"
                     "based_in"     -> "is_based_in"
                     "born_in"      -> "is_born_in"

                Important:
                - NEVER swap subject/object for relations ending with: "_in", "_on", "_at", "_of"
                - Maintain all extra tuple fields exactly as in input
                - Do NOT generate new tuples

                Output Format (STRICT JSON):
                Return a JSON object with key "normalized_triple", containing a list of normalized tuples.
                Keep the original tuple string format; only adjust relation and subject/object if needed.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱关系方向标准化专家。

                任务：
                - 将关系统一为规范的 "is_*" 形式。
                - 仅在施事型动作关系中交换主语和宾语。
                - 输入元组可能包含三元组、四元组或更多元素（如主语、宾语、关系、时间、上下文）。
                - 仅修改关系方向或主语/宾语顺序，其他字段保持不变。

                关系处理规则：
                1. 施事型动作关系（需交换主语/宾语）：
                   示例：
                     "develops" -> "is_developed_by"
                     "creates"  -> "is_created_by"
                     "sponsors" -> "is_sponsored_by"
                     "produces" -> "is_produced_by"
                2. 状态/位置/属性关系（不交换主语/宾语）：
                   示例：
                     "performed_in" -> "is_performed_in"
                     "located_in"   -> "is_located_in"
                     "based_in"     -> "is_based_in"
                     "born_in"      -> "is_born_in"

                注意事项：
                - 以 "_in"、"_on"、"_at"、"_of" 结尾的关系绝对禁止交换主语/宾语
                - 保留所有额外字段
                - 不生成新元组

                输出格式（严格 JSON）：
                返回 JSON 对象，键名为 "normalized_triple"，值为规范化后的元组列表。
                保持输入元组字符串格式，仅调整关系及主语/宾语顺序。
            """)

    def build_prompt(self, tuples: str):
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Normalize the relation directions of the following knowledge graph tuples.
                Input may contain triples, quadruples, or higher-order tuples.
                Keep all extra fields intact, only adjust relation direction or swap subject/object when required.

                Input Tuples:
                {tuples}

                Return ONLY a JSON object with key "normalized_triple".
            """)
        else:
            return textwrap.dedent(f"""\
                对以下知识图谱元组进行关系方向归一化。
                元组可能是三元组、四元组或更多元素。
                保留所有额外字段，仅在必要时修改关系方向或交换主语/宾语。

                输入元组：
                {tuples}

                仅返回 JSON 对象，键名为 "normalized_triple"。
            """)



@PROMPT_REGISTRY.register()
class KGEntityRelationTripleDisambiguationPrompt(PromptABC):
    """
    Dedicated prompt for disambiguating entity–relation–entity triples.
    Only resolves ambiguity in relations or tail entities.
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
                You are an expert in knowledge graph entity–relation triple disambiguation.

                Task:
                - Input triples may contain ambiguity in relation names or tail entities,
                  represented by multiple candidates separated by "|" (pipe).
                - Select the single most correct, standard, and widely accepted relation triple.
                - Use commonsense and real-world knowledge when necessary.

                Input triple format:
                "<entity> HeadEntity <relation> RelationName <entity> TailEntity1 | TailEntity2 | ..."

                Rules:
                1. Keep the triple structure unchanged.
                2. Select ONLY ONE relation–entity pair for each input triple.
                3. Do NOT modify the head entity.
                4. Do NOT add explanations, comments, or additional triples.
                5. Output must be valid JSON.

                Example:
                Input:
                "<entity> Henry <relation> affiliated_with <entity> Maple Leafs | Toronto Maple Leafs"

                Output:
                {
                  "resolved_relation": [
                    "<entity> Henry <relation> affiliated_with <entity> Toronto Maple Leafs"
                  ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱实体关系三元组消岐专家。

                任务：
                - 输入的实体关系三元组中，关系名或尾实体可能存在多个候选项，
                  候选项之间用 "|" 分隔。
                - 选择最合理、最标准、最符合常识的一条实体关系三元组。
                - 可以使用现实世界常识进行判断。

                输入三元组格式：
                "<entity> 头实体 <relation> 关系名 <entity> 尾实体1 | 尾实体2 | ..."

                规则：
                1. 保持三元组结构不变。
                2. 每个输入三元组只输出一条消岐后的结果。
                3. 不修改头实体。
                4. 不添加任何解释或额外信息。
                5. 输出必须是合法 JSON。

                示例：
                输入：
                "<entity> Henry <relation> affiliated_with <entity> Maple Leafs | Toronto Maple Leafs"

                输出：
                {
                  "resolved_relation": [
                    "<entity> Henry <relation> affiliated_with <entity> Toronto Maple Leafs"
                  ]
                }
            """)

    # --------------------------------------------------
    # User Prompt
    # --------------------------------------------------
    def build_prompt(self, ambiguous_relation_triple: str):
        """
        Build a prompt for disambiguating entity–relation triples.

        Args:
            ambiguous_relation_triple (str):
                An entity–relation–entity triple with ambiguous candidates.

        Returns:
            str: Prompt ready for LLM inference.
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Disambiguate the following entity–relation triple.
                Select the single most correct relation and tail entity
                when multiple candidates are provided.

                Ambiguous Relation Triple:
                {ambiguous_relation_triple}

                Return ONLY a JSON object with key "resolved_relation"
                and value as a list containing the resolved triple.
            """)
        else:
            return textwrap.dedent(f"""\
                对以下实体关系三元组进行消岐。
                当关系或尾实体存在多个候选项时，选择最合理的一项。

                输入实体关系三元组：
                {ambiguous_relation_triple}

                仅返回 JSON 对象，键名为 "resolved_relation"，值为消岐后的三元组列表。
            """)
