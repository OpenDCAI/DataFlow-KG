from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
import pandas as pd
import re
import json
from typing import List

from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.application_kg.graph_reasoning import KGReasoningRelationInferencePrompt

@prompt_restrict(
    KGReasoningRelationInferencePrompt
)
@OPERATOR_REGISTRY.register()
class KGReasoningRelationGeneration(OperatorABC):
    """
    Infer relations between target entity pairs using LLM based on KG paths.

    Input columns:
        - target_entity: List[List[str]]  e.g. [["Henry, Berlin"], ["Henry, Rome"]]
        - mpath: List[List[List[str]]]    paths corresponding to each entity pair

    Output columns:
        - inferred_triplets: List[List[str]]  # inferred relation triples per entity pair
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        restrict_to_path_rel: bool = True,
        lang: str = "en"
    ):
        self.llm_serving = llm_serving
        self.restrict_to_path_rel = restrict_to_path_rel
        self.lang = lang
        self.logger = get_logger()
        self.rel_pattern = re.compile(r"<subj>\s*(.+?)\s*<obj>\s*(.+?)\s*<rel>\s*(.+)$")
        self.prompt = KGReasoningRelationInferencePrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGReasoningRelationGeneration 用于基于多跳路径推断实体关系。",
                "输入: target_entity + mpath; 输出: inferred_triplets",
            )
        return (
            "KGReasoningRelationGeneration is used to infer entity relations from multi-hop paths.",
            "Input: target_entity + mpath; Output: inferred_triplets",
        )

    # ----------------------------
    # DataFrame validation
    # ----------------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = ["target_entity", "mpath"]
        missing = [k for k in required_keys if k not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

    # ----------------------------
    # Extract subj/obj directly from target_entity
    # ----------------------------
    def _extract_entities_from_target(self, target_pair_raw: list):
        """
        target_pair_raw: e.g. ["Henry, Berlin"]
        Returns: subj, obj
        """
        if not target_pair_raw:
            return None, None
        text = target_pair_raw[0] if isinstance(target_pair_raw, list) else target_pair_raw
        parts = [t.strip() for t in text.split(",") if t.strip()]
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]

    # ----------------------------
    # Collect candidate relations from paths
    # ----------------------------
    def _collect_candidate_rels(self, pair_paths: List[List[str]]):
        candidate_rels = set()
        if self.restrict_to_path_rel:
            for path in pair_paths:
                for triple in path:
                    m = self.rel_pattern.search(triple)
                    if m:
                        candidate_rels.add(m.group(3).strip())
        return list(candidate_rels)

    # ----------------------------
    # Call LLM for one entity pair
    # ----------------------------
    def _infer_for_pair(self, subj: str, obj: str, pair_paths: List[List[str]]):
        candidate_rels = self._collect_candidate_rels(pair_paths)
        try:
            user_inputs = [
                self.prompt.build_prompt(subj, obj, pair_paths, candidate_rels)
            ]
            sys_prompt = self.prompt.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs, system_prompt=sys_prompt
            )
            # 解析 LLM 输出 JSON 数组
            raw = responses[0]
            try:
                inferred = json.loads(raw)
                if not isinstance(inferred, list):
                    inferred = []
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM output for {subj}-{obj}: {raw} | {e}")
                inferred = []
        except Exception as e:
            self.logger.warning(f"LLM call failed for pair {subj}-{obj}: {e}")
            inferred = []
        return inferred

    # ----------------------------
    # Run
    # ----------------------------
    def run(
        self,
        storage: DataFlowStorage,
        target_key: str = "target_entity",
        path_key: str = "mpath",
        output_key: str = "inferred_triplets"
    ) -> List[str]:

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        all_inferred = []

        for _, row in df.iterrows():
            entity_pairs = row[target_key]           # e.g. [["Henry, Berlin"], ["Henry, Rome"]]
            paths_per_pair = row[path_key]           # e.g. [[["triple1",...], [...]], [...]]
            inferred_for_row = []

            for idx, pair_raw in enumerate(entity_pairs):
                subj, obj = self._extract_entities_from_target(pair_raw)
                if subj is None or obj is None:
                    inferred_for_row.append([])
                    continue
                pair_paths = paths_per_pair[idx] if idx < len(paths_per_pair) else []
                inferred = self._infer_for_pair(subj, obj, pair_paths)
                inferred_for_row.append(inferred)

            all_inferred.append(inferred_for_row)

        df[output_key] = all_inferred
        storage.write(df)
        self.logger.info(f"Inferred relations saved to column {output_key}")
        return [output_key]
