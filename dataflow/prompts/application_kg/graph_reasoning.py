import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json
from typing import List, Optional

@PROMPT_REGISTRY.register()
class KGReasoningRelationInferencePrompt(PromptABC):
    """
    Prompt: 根据目标实体对和已搜索的多跳路径，推断可能关系。

    - 输入：
        * 一个实体对 (subj, obj)
        * 该实体对对应的多跳路径 (每条路径是三元组列表)
        * 可选的候选关系列表（restrict_to_path_rel=True 时提供）
    - 输出：
        * 推断出的三元组列表，每个三元组格式与输入一致：
          <subj> ... <obj> ... <rel> ...
    - 用途：
        * KGPathRelationInferenceLLM Operator 调用 LLM 推断实体对关系
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self):
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a Knowledge Graph reasoning expert.

                === TASK ===
                Given:
                - An entity pair: Subj and Obj
                - Multi-hop paths connecting them (each path is a list of KG triples)
                - Optional candidate relations

                Infer plausible KG triples that connect the entity pair.
                Each triple must follow the format: <subj> ... <obj> ... <rel> ...

                === STRICT RULES ===
                - Only generate triples for the given entity pair
                - You may select relations from candidate_relations if provided
                - If no candidate relations are given, there is no restriction and you may freely generate plausible relations
                - Do NOT invent entities
                - Do NOT explain reasoning
                - Output all plausible relations (multi-hop reasoning allowed)
                - Do NOT include extra text

                === OUTPUT FORMAT (STRICT JSON) ===
                [
                  "<subj> Entity1 <obj> Entity2 <rel> relation1",
                  "<subj> Entity1 <obj> Entity2 <rel> relation2"
                ]

                JSON array only. No additional text.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱推理专家。

                === 任务 ===
                已知：
                - 一个实体对：Subj 和 Obj
                - 连接该实体对的多跳路径（每条路径是三元组列表）
                - 可选候选关系列表

                推断出该实体对可能存在的知识图谱三元组。
                每个三元组必须遵循格式：<subj> ... <obj> ... <rel> ...

                === 严格规则 ===
                - 仅生成该实体对的三元组
                - 可以从候选关系中选择，如果提供
                - 如果没有提供候选关系，则可以自由生成关系
                - 不允许臆造实体
                - 不输出推理过程
                - 尽量输出所有可能的关系（三跳或多跳均可）
                - 不输出多余文字

                === 输出格式（严格 JSON）===
                [
                  "<subj> 实体1 <obj> 实体2 <rel> 关系1",
                  "<subj> 实体1 <obj> 实体2 <rel> 关系2"
                ]

                仅输出 JSON 数组，不要输出额外文本。
            """)

    def build_prompt(self, subj: str, obj: str, paths: List[List[str]], candidate_rels: List[str]):
        """
        构建 LLM 输入，paths 是多跳路径三元组列表，candidate_rels 是候选关系列表
        """
        paths_text = "\n".join([t for path in paths for t in path])
        candidate_text = ", ".join(candidate_rels) if candidate_rels else "no restriction"

        if self.lang == "en":
            return textwrap.dedent(f"""\
                Entity pair:
                Subj: {subj}
                Obj: {obj}

                Multi-hop paths:
                {paths_text}

                Candidate relations: {candidate_text}

                Please infer all plausible triples connecting the entity pair.
                Output strictly in JSON array format as instructed in system prompt.
            """)
        else:
            return textwrap.dedent(f"""\
                实体对：
                Subj: {subj}
                Obj: {obj}

                多跳路径：
                {paths_text}

                候选关系：{candidate_text}

                请推断出该实体对可能存在的所有三元组。
                严格按照系统提示的 JSON 数组格式输出。
            """)


@PROMPT_REGISTRY.register()
class KGReasoningPathRedundancyPrompt(PromptABC):
    """
    Prompt: 对给定的多跳路径进行冗余程度评分（连续分数 0-1）。

    - 输入：
        * 一个实体对 (subj, obj)
        * 该实体对对应的多跳路径 (每条路径是三元组列表)
    - 输出：
        * 每条路径的冗余度分数 (浮点数 0-1)
          0 = 完全不冗余（每条信息独立）
          1 = 高度冗余（路径中信息重复或多余）
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang.lower()
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are an expert in Knowledge Graph path analysis.

                === TASK ===
                Given a set of multi-hop paths connecting an entity pair, 
                evaluate the redundancy of each path.

                Redundancy criteria:
                - 0 = no redundancy, all triples in the path provide unique information
                - 1 = highly redundant, path contains repeated or unnecessary triples

                Use **continuous scores** between 0 and 1 (e.g., 0.0, 0.25, 0.73, 1.0)
                reflecting the degree of redundancy.

                Only output JSON:
                {
                    "redundancy_scores": [float, float, ...]
                }

                Each score corresponds to one path in order.
                Do NOT provide explanations or extra text.
            """)
        else:
            return textwrap.dedent("""\
                你是一名知识图谱路径分析专家。

                === 任务 ===
                已知一个实体对及其多跳路径，请对每条路径的冗余程度进行评分。

                冗余评分标准：
                - 0 = 无冗余，路径中每条三元组提供独立信息
                - 1 = 高度冗余，路径中存在重复或多余三元组

                使用 **连续浮点数** 0-1（例如 0.0, 0.25, 0.73, 1.0）表示冗余程度。

                输出仅为 JSON：
                {
                    "redundancy_scores": [float, float, ...]
                }

                每个分数对应输入路径中的一条路径，按顺序输出。
                不要输出解释或额外文字。
            """)

    def build_prompt(self, subj: str, obj: str, paths: List[List[str]]) -> str:
        """
        构建 LLM 输入，paths 是多跳路径三元组列表
        """
        paths_text = ""
        for idx, path in enumerate(paths):
            path_block = "\n".join(path)
            paths_text += f"Path {idx}:\n{path_block}\n\n"

        if self.lang == "en":
            return textwrap.dedent(f"""\
                Entity pair:
                Subj: {subj}
                Obj: {obj}

                Multi-hop paths:
                {paths_text}

                Evaluate the redundancy of each path using **continuous scores 0-1**.
                Return strictly JSON as instructed.
            """)
        else:
            return textwrap.dedent(f"""\
                实体对：
                Subj: {subj}
                Obj: {obj}

                多跳路径：
                {paths_text}

                请对每条路径的冗余程度进行评分，使用 **连续浮点数 0-1**。
                严格按照系统提示的 JSON 输出。
            """)