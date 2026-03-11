"""
====================================
DataFlow-KG:
====================================

Author: Wanpeng Tang
Affiliation: UESTC
Email: 2023090910014@std.uestc.edu.cn
Created: 2026-02-23

Author: Zhengpin
Refined: 2026-03-04

License:
    MIT License
"""

import json
import random
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.core_kg.rel_triple_eval import KGHallucinationEvaluationPrompt


class KGTripleHallucinationEvaluator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        sample_size: int = 10
    ):
        super().__init__()
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be LLMServingABC")

        self.llm_serving = llm_serving
        self.prompt_manager = KGHallucinationEvaluationPrompt(lang)
        self.sample_size = sample_size

    # ============================================================
    # Triple Type Detection
    # ============================================================

    def _is_relation(self, triple: str) -> bool:
        return "<subj>" in triple and "<obj>" in triple and "<rel>" in triple

    def _is_attribute(self, triple: str) -> bool:
        return "<entity>" in triple and "<attribute>" in triple and "<value>" in triple

    # ============================================================
    # Triple Parsing
    # ============================================================

    def _parse_relation(self, triple: str) -> List[str]:
        try:
            subj = triple.split("<subj>")[1].split("<obj>")[0].strip()
            obj = triple.split("<obj>")[1].split("<rel>")[0].strip()
            rel = triple.split("<rel>")[1].strip()
            return [subj, rel, obj]
        except:
            return []

    def _parse_attribute(self, triple: str) -> List[str]:
        try:
            entity = triple.split("<entity>")[1].split("<attribute>")[0].strip()
            attribute = triple.split("<attribute>")[1].split("<value>")[0].strip()
            value = triple.split("<value>")[1].strip()
            return [entity, attribute, value]
        except:
            return []

    def _parse_triples(self, triples: List[str]) -> List[List[str]]:
        parsed = []

        if not triples:
            return parsed

        # 自动判断类型（只看第一个）
        first = triples[0]
        if self._is_relation(first):
            parser = self._parse_relation
        elif self._is_attribute(first):
            parser = self._parse_attribute
        else:
            return parsed

        for t in triples:
            if isinstance(t, str):
                p = parser(t)
                if p:
                    parsed.append(p)

        return parsed

    # ============================================================
    # JSON + Metrics
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        clean = response.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)
        except:
            return {"verifications": []}

    def _calculate_hallucination_rate(self, verifications: List[Dict[str, Any]]) -> float:
        total = len(verifications)
        if total == 0:
            return 0.0

        supported = sum(
            1 for v in verifications
            if v.get("status", "").lower() == "supported"
        )

        return float(1 - supported / total)

    # ============================================================
    # Core Evaluation
    # ============================================================

    def process_batch(self, records: List[Dict[str, Any]]):

        results = []

        for row in tqdm(records, desc="Hallucination Eval"):

            text = row.get("raw_chunk", "")
            test_triples = row.get("test_triple", None)
            triples = row.get("triple", [])

            if isinstance(test_triples, str):
                try:
                    test_triples = json.loads(test_triples)
                except:
                    test_triples = None

            if isinstance(triples, str):
                try:
                    triples = json.loads(triples)
                except:
                    triples = []

            if not text:
                results.append({"hallucination_rate": 0.0})
                continue

            # ====================================================
            # Case 1: Use test_triple (no sampling)
            # ====================================================
            if test_triples:

                parsed = self._parse_triples(test_triples)
                if not parsed:
                    results.append({"hallucination_rate": 0.0})
                    continue

                return_indices = False

            # ====================================================
            # Case 2: Random sample from triple
            # ====================================================
            else:

                if not triples:
                    results.append({
                        "hallucination_rate": 0.0,
                        "eval_sample_indices": []
                    })
                    continue

                total = len(triples)

                if total <= self.sample_size:
                    sample_indices = list(range(total))
                else:
                    sample_indices = random.sample(range(total), self.sample_size)

                sampled = [triples[i] for i in sample_indices]
                parsed = self._parse_triples(sampled)

                return_indices = True

                if not parsed:
                    results.append({
                        "hallucination_rate": 0.0,
                        "eval_sample_indices": sample_indices
                    })
                    continue

            # ====================================================
            # LLM Evaluation
            # ====================================================

            try:
                sys_prompt = self.prompt_manager.build_system_prompt()
                user_prompt = self.prompt_manager.build_prompt(text, parsed)

                response = self.llm_serving.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=sys_prompt
                )[0]

                data = self._safe_parse_json(response)
                verifications = data.get("verifications", [])

                rate = self._calculate_hallucination_rate(verifications)

                if return_indices:
                    results.append({
                        "hallucination_rate": rate,
                        "eval_sample_indices": sample_indices
                    })
                else:
                    results.append({
                        "hallucination_rate": rate
                    })

            except Exception as e:
                self.logger.error(f"LLM Error: {e}")

                if return_indices:
                    results.append({
                        "hallucination_rate": 0.0,
                        "eval_sample_indices": sample_indices
                    })
                else:
                    results.append({"hallucination_rate": 0.0})

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "raw_chunk",
        input_key_meta1: str = "triple",
        input_key_meta2: str = "test_triple"
    ):

        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []
        for _, r in df.iterrows():
            records.append({
                "raw_chunk": r.get(input_key, ""),
                "triple": r.get(input_key_meta1, []),
                "test_triple": r.get(input_key_meta2, None)
            })

        outputs = self.process_batch(records)

        df["hallucination_rate"] = [
            o.get("hallucination_rate", 0.0)
            for o in outputs
        ]

        df["eval_sample_indices"] = [
            o.get("eval_sample_indices", None)
            for o in outputs
        ]

        out_file = storage.write(df)

        self.logger.info(
            f"Avg Hallucination Rate: "
            f"{df['hallucination_rate'].mean():.4f}. "
            f"Saved to {out_file}"
        )

        return ["hallucination_rate", "eval_sample_indices"]