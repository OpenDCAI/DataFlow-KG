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