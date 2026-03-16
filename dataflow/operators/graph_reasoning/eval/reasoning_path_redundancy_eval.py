# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KG Multi-Hop Path Redundancy Evaluator
====================================
Author: Zhengpin Li
Refined: 2026-03-16

License:
    MIT License
"""

import json
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.application_kg.graph_reasoning import KGReasoningPathRedundancyPrompt


class KGPathRedundancyEvaluator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "zh"
    ):
        super().__init__()
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be LLMServingABC")

        self.llm_serving = llm_serving
        self.prompt_manager = KGReasoningPathRedundancyPrompt(lang)

    # ============================================================
    # JSON Parse
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        clean = response.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)
        except:
            self.logger.warning(f"Failed to parse JSON: {response}")
            return {"redundancy_scores": []}

    # ============================================================
    # Core Evaluation
    # ============================================================

    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        for row in tqdm(records, desc="KG Path Redundancy Eval"):

            mpaths = row.get("mpath", [])
            target_entities = row.get("target_entity", [])

            if not mpaths or not isinstance(mpaths, list):
                results.append({"redundancy_scores": []})
                continue

            all_scores = []

            for idx, paths_for_pair in enumerate(mpaths):
                if not isinstance(paths_for_pair, list):
                    all_scores.append([])
                    continue

                # 从 target_entity 中获取 subj 和 obj
                subj, obj = "unknown_subj", "unknown_obj"
                try:
                    pair_raw = target_entities[idx]
                    if isinstance(pair_raw, list) and pair_raw:
                        pair_items = pair_raw[0].split(",")
                        if len(pair_items) == 2:
                            subj, obj = pair_items[0].strip(), pair_items[1].strip()
                except Exception as e:
                    self.logger.warning(f"Failed to extract subj/obj from target_entity: {e}")

                try:
                    system_prompt = self.prompt_manager.system_text
                    user_prompt = self.prompt_manager.build_prompt(subj, obj, paths_for_pair)

                    response = self.llm_serving.generate_from_input(
                        user_inputs=[user_prompt],
                        system_prompt=system_prompt
                    )[0]

                    data = self._safe_parse_json(response)
                    scores = data.get("redundancy_scores", [])

                    all_scores.append(scores)

                except Exception as e:
                    self.logger.error(f"LLM Error: {e}")
                    all_scores.append([])

            results.append({"redundancy_scores": all_scores})

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "mpath",
        target_key: str = "target_entity",
        output_key: str = "redundancy_scores"
    ) -> List[str]:

        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        # 组织记录
        records = []
        for _, r in df.iterrows():
            records.append({
                input_key: r.get(input_key, []),
                target_key: r.get(target_key, [])
            })

        outputs = self.process_batch(records)

        # 保存回 dataframe
        df[output_key] = [
            o.get(output_key, [])
            for o in outputs
        ]

        out_file = storage.write(df)
        self.logger.info(f"Saved KG path redundancy scores to {out_file}")

        return [output_key]