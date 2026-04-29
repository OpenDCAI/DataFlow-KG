"""
====================================
DataFlow-KG:
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()
class KGRelationStrengthScoringPrompt(PromptABC):
    """
    专属 Prompt：根据文本和三元组，对每条三元组关系强度进行打分
    输入：
        - source_texts: 文本内容
        - extracted_triples: 已抽取的三元组列表
    输出：
        - 每条三元组的关系强度分数列表 [0,1]
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph evaluator.

                Task:
                - Given a source text and a list of extracted knowledge graph triples,
                  assign a confidence score to each triple indicating the strength
                  or reliability of the relationship.
                - Each score should be a number between 0 and 1.
                - 0 means very weak/uncertain relation; 1 means very strong/well-supported relation.

                Rules:
                1. Base your scores on textual evidence, plausibility, and common sense.
                2. Do NOT modify the triples.
                3. Output only a list of floats corresponding to each triple.

                Output format (strict JSON):
                {{
                "triple_strength_score": [0.82, 0.7, 0.95, ...]
                }}
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱评测专家。

                任务：
                - 根据给定的文本内容和已抽取的三元组列表，
                  为每条三元组打关系强度分数。
                - 分数范围 [0,1]，0 表示关系很弱或不确定，1 表示关系很强或高度可信。

                规则：
                1. 根据文本证据、合理性和常识评估关系强度。
                2. 不要修改三元组内容。
                3. 输出与三元组列表一一对应的浮点数列表。

                输出格式（严格 JSON）：
                {{
                "triple_strength_score": [0.82, 0.7, 0.95, ...]
                }}
            """)

    def build_prompt(self, source_texts: str, extracted_triples: str):
        """
        构建 prompt，输入文本和三元组
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please evaluate the relation strength of each triple based on the source text.

                Source Texts:
                {source_texts}

                Extracted Triples:
                {extracted_triples}

                Output STRICT JSON only:
            """)
        else:
            return textwrap.dedent(f"""\
                请根据文本内容评估每条三元组的关系强度。

                来源文本：
                {source_texts}

                已抽取三元组：
                {extracted_triples}

                严格 JSON 输出：
            """)


@PROMPT_REGISTRY.register()
class KGTripleAccuracyEvaluatorPrompt(PromptABC):
    """
    专属 Prompt：评测给定文本的三元组准确性

    输入：
        - source_texts: 文本内容
        - extracted_triples: 已抽取的知识图谱三元组

    输出：
        - 正确三元组比例
        - 不准确或错误三元组列表
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a knowledge graph evaluator and judge.

                Task:
                - Evaluate the accuracy of the provided knowledge graph triples 
                  against the source texts.
                - A triple is accurate if it correctly represents information 
                  explicitly stated in the text.
                - Identify incorrect or partially incorrect triples.
                - Provide quantitative metrics about accuracy and a list of incorrect triples.
                - Accuracy_score must be a number between 0 and 1 representing the fraction of correct triples.

                Evaluation rules:
                1. Accurate triple: the triple correctly expresses a fact, relation, or event in the text.
                2. Inaccurate triple: the triple is inconsistent with the text, contains hallucinated information, or misrepresents the fact.
                3. Accuracy score: number of accurate triples / total triples.

                Output format (strict JSON):
                {
                  "accuracy_score": score,  # fraction of accurate triples
                  "incorrect_triples": [
                    <subj> HeadEntity <obj> TailEntity <rel> Relation,
                    <subj> HeadEntity <obj> TailEntity <rel> Relation
                  ]
                }
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱评测专家。

                任务：
                - 根据文本内容，评估提供的知识图谱三元组的准确性。
                - 如果三元组准确地表达了文本中的事实或关系，则认为正确。
                - 识别不准确或错误的三元组。
                - 输出量化指标与不准确三元组列表。

                评测规则：
                1. 正确三元组：三元组准确表达文本中的事实、关系或事件。
                2. 不准确三元组：三元组与文本不一致，包含虚假信息或曲解事实。
                3. 准确率：准确三元组数量 / 总三元组数量。

                输出格式（严格 JSON）：
                {
                  "accuracy_score": score,  # 正确三元组比例
                  "incorrect_triples":[
                    <subj> HeadEntity <obj> TailEntity <rel> Relation,
                    <subj> HeadEntity <obj> TailEntity <rel> Relation
                  ]
                }
            """)

    def build_prompt(
        self,
        source_texts: str,
        extracted_triples: str
    ):
        """
        构建评测型 prompt

        Args:
            source_texts: 文本内容
            extracted_triples: 已抽取的知识图谱三元组

        Returns:
            str: 可直接发送给 LLM 的 prompt
        """
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Please evaluate the accuracy of the following extracted triples against the source texts.

                Source Texts:
                {source_texts}

                Extracted Triples:
                {extracted_triples}

                Output STRICT JSON only with:
                - accuracy_score
                - incorrect_triples
            """)
        else:
            return textwrap.dedent(f"""\
                请评测以下已抽取的三元组与文本内容的一致性和准确性。

                来源文本：
                {source_texts}

                已抽取三元组：
                {extracted_triples}

                输出严格 JSON，仅包含：
                - accuracy_score
                - incorrect_triples
            """)



@PROMPT_REGISTRY.register()  # pyright: ignore[reportOptionalCall]
class KGRelationConsistencyEvaluationPrompt(PromptABC):
    """
    Prompt for Knowledge Graph Consistency Evaluation via Contextual Inference.
    Uses masked relation prediction to assess logical consistency.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang

    # ============================================================
    # System Prompt
    # ============================================================
    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名知识图谱审计专家。你的任务是根据给定的上下文信息，
                判断目标三元组在逻辑和语义上是否一致。

                【评估原则】

                1. 逻辑一致性
                   目标三元组不能与上下文事实产生冲突。
                   包括显式冲突（如不同出生年份）和隐式冲突（如互斥身份）。

                2. 语义类型匹配
                   关系必须符合主体与客体的语义类型与常识。
                   若关系明显违反实体类型或现实世界知识，应判为 INCONSISTENT。

                3. 开放世界假设
                   若上下文未提供直接支持，但也不存在冲突，
                   只要关系在语义上合理，应判为 CONSISTENT。
                   缺乏证据不等于不一致。

                4. 保守判错原则
                   只有在存在明确逻辑冲突或严重语义错误时，
                   才可判为 INCONSISTENT。
                   若存在不确定性但无冲突，应默认判为 CONSISTENT。

                【输出格式】
                你必须输出一个 JSON 对象：
                {
                    "judgment": "CONSISTENT" 或 "INCONSISTENT"
                }
            """)
        else:
            return textwrap.dedent("""\
                You are a Knowledge Graph Auditor. Your task is to evaluate whether 
                a given triple is logically and semantically consistent 
                with its surrounding context.

                ### Evaluation Principles

                1. Logical Non-Contradiction
                   The target triple must not contradict any fact in the context.
                   This includes explicit contradictions (e.g., different birth years)
                   and implicit contradictions (e.g., mutually exclusive roles).

                2. Semantic Type Compatibility
                   The relation must be compatible with the semantic types 
                   of the subject and object.
                   Clearly invalid type combinations must be judged as INCONSISTENT.

                3. Open-World Assumption
                   The absence of supporting evidence does NOT imply inconsistency.
                   If the relation is plausible and not contradictory,
                   it should be judged as CONSISTENT.

                4. Conservative Inconsistency Policy
                   Only output INCONSISTENT when there is clear logical contradiction
                   or strong semantic violation.
                   If uncertain but no contradiction exists, default to CONSISTENT.

                ### Output Format
                You must output a single JSON object:
                {
                    "judgment": "CONSISTENT" or "INCONSISTENT"
                }
            """)

    # ============================================================
    # User Prompt
    # ============================================================
    def build_prompt(
        self,
        context_desc: str,
        subj: str,
        obj: str,
        relation: str
    ) -> str:

        if self.lang == "zh":
            return textwrap.dedent(f"""\
                【上下文（邻居事实）】
                {context_desc}

                【待验证三元组】
                - 主体: "{subj}"
                - 客体: "{obj}"
                - 关系: "{relation}"

                请根据上述上下文判断该三元组是 CONSISTENT 还是 INCONSISTENT。
                按照规定的 JSON 格式输出结果。
            """)
        else:
            return textwrap.dedent(f"""\
                ### Context (Neighboring Facts)
                {context_desc}

                ### Target Triple to Verify
                - Subject: "{subj}"
                - Object: "{obj}"
                - Relation: "{relation}"

                Based on the context above,
                determine whether the target triple is CONSISTENT or INCONSISTENT.
                Output your judgment strictly in the required JSON format.
            """)



@PROMPT_REGISTRY.register()  # pyright: ignore[reportOptionalCall]
class KGHallucinationEvaluationPrompt(PromptABC):
    """
    Unified Semantic NLI Prompt for Hallucination Detection.
    Supports both English and Chinese. 
    Focuses on semantic entailment rather than exact string matching to handle aliases and pronouns.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名严格的知识图谱事实校验审判员。
                你的任务是验证提取出的三元组是否**忠实地被源文本支持**。

                ### 判断类别
                1. **支持 (Supported)**：三元组的含义在文本中明确陈述或强烈暗示。(允许同义词或代词)
                2. **矛盾 (Contradicted)**：三元组与文本内容冲突。例如，文本说“A 喜欢 B”，而三元组说“A 讨厌 B”。
                3. **未提及 (Not Mentioned)**：文本中完全没有相关信息。例如，三元组关于“马斯克”，但文本谈论的是“乔布斯”。

                ### 实体对应注意事项
                - 不要只寻找精确字符串匹配。
                - 如果文本说“创始人”，三元组说“史蒂夫·乔布斯”，且语境能对应 -> 判定为 **支持**。
                - 如果文本使用代词“他”，三元组使用具体名字 -> 判定为 **支持**（前提是代词解析正确）。

                ### 输出格式
                仅返回一个 JSON 对象：
                {
                    "verifications": [
                        {
                            "triple_id": <int>,
                            "status": "Supported" | "Contradicted" | "Not Mentioned"
                        },
                        ...
                    ]
                }
            """)
        else:
            return textwrap.dedent("""\
                You are a rigorous Fact-Checking Judge for Knowledge Graphs.
                Your task is to verify if extracted Triples are **faithfully supported** by the Source Text.

                ### Judgment Categories
                1. **Supported**: The triple's meaning is explicitly stated or strongly implied by the text. (Synonyms/Pronouns are OK)
                2. **Contradicted**: The triple says something that conflicts with the text (e.g., text says "A likes B", triple says "A hates B")
                3. **Not Mentioned**: The information is completely absent (e.g., Triple is about "Elon Musk", but text is about "Steve Jobs")

                ### Important: Entity Grounding
                - Do NOT look for exact string matches.
                - If text says "The founder", and triple says "Steve Jobs", AND the context implies they are the same -> This is **Supported**.
                - If text says "He", and triple uses the specific name -> This is **Supported** (if the resolution is correct).

                ### Output Format
                Return ONLY a JSON object:
                {
                    "verifications": [
                        {
                            "triple_id": <int>,
                            "status": "Supported" | "Contradicted" | "Not Mentioned"
                        },
                        ...
                    ]
                }
            """)

    def build_prompt(self, text: str, triples: list) -> str:
        triples_block = ""
        for idx, t in enumerate(triples):
            # Format: ID: <Subject> <Predicate> <Object>
            triples_block += f"ID {idx}: <{t[0]}> <{t[1]}> <{t[2]}>\n"

        if self.lang == "zh":
            return f"""请根据下列源文本验证三元组的真实性。

            --- 源文本 ---
            {text}

            --- 待验证三元组 ---
            {triples_block}

            请仅返回 JSON 格式的评估结果。"""
        else:
            return f"""Verify these triples against the source text.

            --- Source Text ---
            {text}

            --- Triples to Verify ---
            {triples_block}

            Provide the JSON evaluation."""


@PROMPT_REGISTRY.register()
class KGSubgraphConsistencyPrompt(PromptABC):
    """
    Judge the internal semantic consistency of a subgraph.
    Accepts input as textual quadruples in the form:
    "<subj> ... <obj> ... <rel> ..."
    Supports English and Chinese.
    Returns a consistency score between 0 (fully inconsistent) and 1 (fully consistent).
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名知识图谱审查专家。
                你的任务是评估一个子图的内部语义一致性。

                ### 判断要求
                - 分析子图中之间是否存在逻辑冲突。
                - 检查同一实体或关系在不同四元组中是否自洽。
                - 考虑别名、代词、上下文语义，不局限于字符串匹配。

                ### 输出
                仅返回一个 JSON 对象：
                {
                    "consistency_score": <float>  // 范围 0-1，1表示完全一致，0表示完全矛盾
                }
                不要输出其他文本或解释。
            """)
        else:
            return textwrap.dedent("""\
                You are an expert in Knowledge Graph evaluation.
                Your task is to assess the **internal semantic consistency** of a subgraph:

                ### Instructions
                - Analyze whether the triples in the subgraph have logical conflicts.
                - Check if the same entities or relations are coherent across quadruples.
                - Consider aliases, pronouns, and semantic context; do NOT rely solely on string matches.

                ### Output
                Return ONLY a JSON object:
                {
                    "consistency_score": <float>  // Range 0-1, 1 means fully consistent, 0 means fully contradictory
                }
                Do not output any explanations.
            """)

    def build_prompt(self, subgraph: list) -> str:
        """
        Format textual quadruples for LLM input and request consistency scoring.

        Args:
            subgraph (list): list of quadruples, each in string form:
                "<subj> ... <obj> ... <rel> ..."
        """
        quadruples_block = ""
        for idx, q in enumerate(subgraph):
            quadruples_block += f"ID {idx}: {q}\n"

        if self.lang == "zh":
            return f"""请评估以下子图的内部一致性。

            --- 子图 ---
            {quadruples_block}

            请仅返回 JSON 格式的一致性得分（0-1）。"""
        else:
            return f"""Assess the internal semantic consistency of the following subgraph.

            --- Subgraph Quadruples ---
            {quadruples_block}

            Return ONLY a JSON object with a consistency score between 0 and 1."""



@PROMPT_REGISTRY.register()
class KGQAConcisenessPrompt(PromptABC):
    """
    Evaluate the conciseness of QA pairs.

    Each QA pair is formatted as:
    {"question": "...", "answer": "..."}

    The model should score each QA pair independently based on how concise
    and direct the answer is.

    Score range:
        0 = very verbose / redundant
        1 = perfectly concise
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名问答质量评估专家。
                你的任务是评估每个问答对的“简洁性”。

                ### 判断标准
                - 回答是否直接回答问题
                - 是否包含多余或冗长信息
                - 是否使用最少的必要词语表达答案
                - 不要因为答案短就直接给高分，需要判断是否刚好回答问题

                ### 输出格式
                仅返回 JSON：
                {
                    "conciseness_scores": [float, float, ...]
                }

                每个 QA 对应一个分数，范围 0-1：
                1 = 非常简洁
                0 = 非常冗余
                不要输出任何解释。
            """)
        else:
            return textwrap.dedent("""\
                You are an expert in QA quality evaluation.
                Your task is to evaluate the **conciseness** of each QA pair.

                ### Evaluation Criteria
                - Does the answer directly respond to the question?
                - Is the answer free of unnecessary information?
                - Is the answer expressed using minimal necessary words?
                - Do NOT give a high score just because the answer is short.

                ### Output Format
                Return ONLY a JSON object:

                {
                    "conciseness_scores": [float, float, ...]
                }

                Each score must correspond to one QA pair in order.
                Score range: 0-1
                1 = very concise
                0 = very verbose or redundant

                Do not output explanations.
            """)

    def build_prompt(self, QA_pairs: list) -> str:
        """
        Format QA pairs for LLM evaluation.

        Args:
            QA_pairs (list): list of QA strings
        """

        qa_block = ""
        for idx, qa in enumerate(QA_pairs):
            qa_block += f"ID {idx}: {qa}\n"

        if self.lang == "zh":
            return f"""请评估以下问答对的回答是否简洁。

            --- QA Pairs ---
            {qa_block}

            请返回每个问答对的简洁性得分（0-1），并严格按照 JSON 输出。"""
        else:
            return f"""Evaluate the conciseness of the answers in the following QA pairs.

            --- QA Pairs ---
            {qa_block}

            Return ONLY a JSON object containing conciseness scores for each QA pair (0-1)."""



@PROMPT_REGISTRY.register()
class KGQACorrelationPrompt(PromptABC):
    """
    Evaluate the correlation between question and answer in QA pairs.

    Each QA pair is formatted as:
        {"question": "...", "answer": "..."}

    The model should determine whether the answer actually responds
    to the question.

    Score range:
        0 = completely unrelated
        1 = perfectly correlated
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名问答质量评估专家。
                你的任务是评估每个问答对中“问题和答案之间的相关性”。

                ### 判断标准
                - 答案是否直接回答了问题
                - 答案是否与问题语义相关
                - 是否存在答非所问的情况
                - 若答案仅部分回答问题，则给中等分数

                ### 输出格式
                仅返回 JSON：
                {
                    "correlation_scores": [float, float, ...]
                }

                每个 QA 对应一个分数，范围 0-1：
                1 = 完全回答问题
                0.5 = 部分相关
                0 = 完全不相关

                不要输出任何解释。
            """)

        else:
            return textwrap.dedent("""\
                You are an expert in QA quality evaluation.
                Your task is to evaluate the **correlation between the question and answer**.

                ### Evaluation Criteria
                - Does the answer actually respond to the question?
                - Is the answer semantically related to the question?
                - Detect cases where the answer does not address the question.
                - Partial answers should receive a medium score.

                ### Output Format
                Return ONLY a JSON object:

                {
                    "correlation_scores": [float, float, ...]
                }

                Each score must correspond to one QA pair in order.

                Score range:
                1 = perfectly answers the question
                0.5 = partially related
                0 = unrelated

                Do not output explanations.
            """)

    def build_prompt(self, QA_pairs: list) -> str:
        """
        Format QA pairs for LLM evaluation.

        Args:
            QA_pairs (list): list of QA strings
        """

        qa_block = ""
        for idx, qa in enumerate(QA_pairs):
            qa_block += f"ID {idx}: {qa}\n"

        if self.lang == "zh":
            return f"""请评估以下问答对中“问题与答案之间的相关性”。

            --- QA Pairs ---
            {qa_block}

            请返回每个问答对的相关性得分（0-1），并严格按照 JSON 输出。"""

        else:
            return f"""Evaluate the correlation between the question and answer in the following QA pairs.

            --- QA Pairs ---
            {qa_block}

            Return ONLY a JSON object containing correlation scores for each QA pair (0-1)."""


@PROMPT_REGISTRY.register()
class KGQANaturalnessPrompt(PromptABC):
    """
    Evaluate the naturalness of QA pairs.

    Each QA pair is formatted as:
        {"question": "...", "answer": "..."}

    The model should judge whether the QA pair sounds natural,
    fluent, and human-like.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang.lower()

    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一名问答质量评估专家。
                你的任务是评估每个问答对的“自然性”。

                ### 判断标准
                - 问题和答案是否符合自然语言表达习惯
                - 是否流畅、易读
                - 是否像人类真实提出的问题和回答
                - 是否存在机械翻译或模板化表达
                - 是否存在语法或表达错误

                ### 输出格式
                仅返回 JSON：
                {
                    "naturalness_scores": [float, float, ...]
                }

                每个 QA 对应一个分数，范围 0-1：
                1 = 非常自然，接近人类表达
                0.5 = 一般自然，有轻微不自然
                0 = 非常不自然或难以理解

                不要输出任何解释。
            """)

        else:
            return textwrap.dedent("""\
                You are an expert in QA quality evaluation.
                Your task is to evaluate the **naturalness** of QA pairs.

                ### Evaluation Criteria
                - Does the question sound like a natural human question?
                - Does the answer sound fluent and natural?
                - Is the QA pair easy to read and grammatically correct?
                - Detect robotic, template-like, or awkward expressions.

                ### Output Format
                Return ONLY a JSON object:

                {
                    "naturalness_scores": [float, float, ...]
                }

                Each score must correspond to one QA pair in order.

                Score range:
                1 = very natural and human-like
                0.5 = somewhat natural
                0 = unnatural or awkward

                Do not output explanations.
            """)

    def build_prompt(self, QA_pairs: list) -> str:
        """
        Format QA pairs for LLM evaluation.

        Args:
            QA_pairs (list): list of QA strings
        """

        qa_block = ""
        for idx, qa in enumerate(QA_pairs):
            qa_block += f"ID {idx}: {qa}\n"

        if self.lang == "zh":
            return f"""请评估以下问答对的自然性。

            --- QA Pairs ---
            {qa_block}

            请返回每个问答对的自然性得分（0-1），并严格按照 JSON 输出。"""

        else:
            return f"""Evaluate the naturalness of the following QA pairs.

            --- QA Pairs ---
            {qa_block}

            Return ONLY a JSON object containing naturalness scores for each QA pair (0-1)."""