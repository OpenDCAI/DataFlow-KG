"""
====================================
DataFlow-KG: QA Conciseness Evaluator
====================================

Author: Wanpeng Tang
Affiliation: UESTC
Email: 2023090910014@std.uestc.edu.cn
Created: 2026-02-23

Author: Zhengpin
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
from dataflow.prompts.core_kg.rel_triple_eval import KGQACorrelationPrompt


class KGQACorrelationEvaluator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en"
    ):
        super().__init__()

        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be LLMServingABC")

        self.llm_serving = llm_serving
        self.prompt_manager = KGQACorrelationPrompt(lang)

    # ============================================================
    # JSON Parse
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:

        clean = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except:
            return {"correlation_scores": []}

    # ============================================================
    # Core Evaluation
    # ============================================================

    def process_batch(self, records: List[Dict[str, Any]]):

        results = []

        for row in tqdm(records, desc="QA Conciseness Eval"):

            qa_pairs = row.get("QA_pairs", [])

            if isinstance(qa_pairs, str):
                try:
                    qa_pairs = json.loads(qa_pairs)
                except:
                    qa_pairs = []

            if not qa_pairs:
                results.append({
                    "correlation_scores": []
                })
                continue

            try:

                system_prompt = self.prompt_manager.build_system_prompt()
                user_prompt = self.prompt_manager.build_prompt(qa_pairs)

                response = self.llm_serving.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=system_prompt
                )[0]

                data = self._safe_parse_json(response)

                scores = data.get("correlation_scores", [])

                results.append({
                    "correlation_scores": scores
                })

            except Exception as e:

                self.logger.error(f"LLM Error: {e}")

                results.append({
                    "correlation_scores": []
                })

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "QA_pairs",
        output_key: str = "correlation_scores"
    ):

        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []

        for _, r in df.iterrows():

            records.append({
                "QA_pairs": r.get(input_key, [])
            })

        outputs = self.process_batch(records)

        df[output_key] = [
            o.get(output_key, [])
            for o in outputs
        ]

        out_file = storage.write(df)

        self.logger.info(
            f"Saved QA conciseness scores to {out_file}"
        )

        return ["correlation_scores"]