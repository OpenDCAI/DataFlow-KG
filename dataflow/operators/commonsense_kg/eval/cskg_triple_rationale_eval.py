"""
====================================
DataFlow-KG: QA Naturalness Evaluator
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
from dataflow.prompts.diverse_kg.cskg import CSKGTripleRationalePrompt


class CSKGTripleRationaleEvaluator(OperatorABC):

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
        self.prompt_manager = CSKGTripleRationalePrompt(lang)

    # ============================================================
    # JSON Parse
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:

        clean = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except:
            return {"naturalness_scores": []}

    # ============================================================
    # Core Evaluation
    # ============================================================

    def process_batch(self, records: List[Dict[str, Any]]):

        results = []

        for row in tqdm(records, desc="QA Naturalness Eval"):

            qa_pairs = row.get("triple", [])

            if isinstance(qa_pairs, str):
                try:
                    qa_pairs = json.loads(qa_pairs)
                except:
                    qa_pairs = []

            if not qa_pairs:
                results.append({
                    "rationale_scores": []
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

                scores = data.get("rationale_scores", [])

                results.append({
                    "rationale_scores": scores
                })

            except Exception as e:

                self.logger.error(f"LLM Error: {e}")

                results.append({
                    "rationale_scores": []
                })

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "rationale_scores"
    ):

        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []

        for _, r in df.iterrows():

            records.append({
                "triple": r.get(input_key, [])
            })

        outputs = self.process_batch(records)

        df[output_key] = [
            o.get(output_key, [])
            for o in outputs
        ]

        out_file = storage.write(df)

        self.logger.info(
            f"Saved QA naturalness scores to {out_file}"
        )

        return [output_key]