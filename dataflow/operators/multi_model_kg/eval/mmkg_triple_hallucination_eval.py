"""
====================================
DataFlow-KG:
====================================

Author: Wanpeng Tang
Affiliation: UESTC
Email: 2023090910014@std.uestc.edu.cn
Created: 2026-02-23

License:
    MIT License
"""

import json
import os
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.prompts.diverse_kg.mmkg import (
    MMKGTextTripleVerificationPrompt,
    MMKGVisualTripleVerificationPrompt,
)


@OPERATOR_REGISTRY.register()  # type: ignore
class MMKGTripleHallucinationEvaluator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        text_llm_serving: Optional[LLMServingABC] = None,
        lang: str = "en",
    ):
        super().__init__()
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be an instance of LLMServingABC")

        if text_llm_serving is not None and not isinstance(text_llm_serving, LLMServingABC):
            raise TypeError("text_llm_serving must be an instance of LLMServingABC")

        self.llm_serving = llm_serving
        self.text_llm_serving = text_llm_serving or llm_serving
        self.lang = lang
        self.is_vlm = self._check_vlm_capability()

        if text_llm_serving is None:
            self.logger.warning(
                "No separate text_llm_serving provided. "
                "Will use llm_serving for text verification."
            )

        self.text_prompt_manager = MMKGTextTripleVerificationPrompt(lang=lang)
        self.visual_prompt_manager = MMKGVisualTripleVerificationPrompt(lang=lang)

    def _check_vlm_capability(self) -> bool:
        service_name = type(self.llm_serving).__name__
        is_vlm = (
            hasattr(self.llm_serving, "generate_from_input_one_image")
            or hasattr(self.llm_serving, "chat_with_one_image")
            or "VLM" in service_name
            or "Vision" in service_name
        )

        if is_vlm:
            self.logger.info(f"Detected VLM service: {service_name}")
        else:
            self.logger.warning(
                f"Service '{service_name}' does not appear to be a VLM. "
                "Visual triple verification may fail."
            )

        return is_vlm

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "多模态知识图谱三元组幻觉评估算子。"
        return "Multi-Modal Knowledge Graph Triple Hallucination Evaluator."

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
        except Exception:
            return []

    def _parse_attribute(self, triple: str) -> List[str]:
        try:
            entity = triple.split("<entity>")[1].split("<attribute>")[0].strip()
            attribute = triple.split("<attribute>")[1].split("<value>")[0].strip()
            value = triple.split("<value>")[1].strip()
            return [entity, attribute, value]
        except Exception:
            return []

    def _parse_visual_relation(self, triple: str) -> List[str]:
        try:
            subj = triple.split("<subj>")[1].split("<rel>")[0].strip()
            rel = triple.split("<rel>")[1].split("<obj>")[0].strip()
            obj = triple.split("<obj>")[1].strip()
            return [subj, rel, obj]
        except Exception:
            return []

    def _parse_triple(self, triple: str) -> Optional[Dict[str, str]]:
        if not isinstance(triple, str):
            return None

        if self._is_relation(triple):
            parts = self._parse_relation(triple)
            if parts:
                return {
                    "type": "relation",
                    "head": parts[0],
                    "edge": parts[1],
                    "tail": parts[2],
                    "raw": triple,
                }

        if self._is_attribute(triple):
            parts = self._parse_attribute(triple)
            if parts:
                return {
                    "type": "attribute",
                    "head": parts[0],
                    "edge": parts[1],
                    "tail": parts[2],
                    "raw": triple,
                }

        return None

    def _parse_visual_triple(self, triple: str) -> Optional[Dict[str, str]]:
        parts = self._parse_visual_relation(triple)
        if not parts:
            return None

        return {
            "type": "visual",
            "head": parts[0],
            "edge": parts[1],
            "tail": parts[2],
            "raw": triple,
        }

    # ============================================================
    # JSON + Metrics
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        clean = response.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)
        except Exception:
            return {}

    def _calculate_hallucination_rate(self, verified: int, hallucinations: int) -> float:
        total = verified + hallucinations
        if total == 0:
            return 0.0
        return float(hallucinations / total)

    def _ensure_list(self, value: Any) -> List[Any]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return []
        return value if isinstance(value, list) else []

    def _ensure_dict(self, value: Any) -> Dict[str, str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return {}
        return value if isinstance(value, dict) else {}

    def _empty_result(self) -> Dict[str, Any]:
        return {"hallucination_rate": 0.0}

    # ============================================================
    # LLM / VLM Calls
    # ============================================================

    def _call_llm(self, user_input: str, system_prompt: str, llm_service: LLMServingABC) -> str:
        try:
            responses = llm_service.generate_from_input(
                user_inputs=[user_input],
                system_prompt=system_prompt,
            )
            if responses:
                return responses[0]
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}", exc_info=True)
        return ""

    def _call_vlm(self, image_path: str, user_prompt: str, system_prompt: str) -> str:
        if not self.is_vlm:
            self.logger.error("VLM service does not support images")
            return ""

        try:
            if hasattr(self.llm_serving, "generate_from_input_one_image"):
                responses = self.llm_serving.generate_from_input_one_image(  # type: ignore
                    image_paths=[image_path],
                    text_prompts=[user_prompt],
                    system_prompt=system_prompt,
                )
                if responses:
                    return responses[0]

            responses = self.llm_serving.generate_from_input(
                user_inputs=[image_path],
                system_prompt=f"{system_prompt}\n\n{user_prompt}",
            )
            if responses:
                return responses[0]
        except Exception as e:
            self.logger.error(f"VLM call failed: {e}", exc_info=True)

        return ""

    # ============================================================
    # Verification
    # ============================================================

    def _verify_text_triple(self, triple: Dict[str, str], context: str) -> Dict[str, Any]:
        system_prompt = self.text_prompt_manager.build_system_prompt()
        user_prompt = self.text_prompt_manager.build_prompt(
            context=context,
            subject=triple["head"],
            relation=triple["edge"],
            obj=triple["tail"],
        )

        response = self._call_llm(user_prompt, system_prompt, self.text_llm_serving)
        if not response:
            return {"reasoning": "Empty LLM response", "verdict": False}

        result = self._safe_parse_json(response)
        if result:
            return {
                "reasoning": result.get("reasoning", ""),
                "verdict": result.get("verdict", False),
            }

        response_upper = response.upper()
        verdict = "TRUE" in response_upper and "FALSE" not in response_upper
        return {"reasoning": response, "verdict": verdict}

    def _verify_visual_triple(
        self,
        triple: Dict[str, str],
        image_path: str,
    ) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {
                "reasoning": f"Image file not found: {image_path}",
                "verdict": "uncertain",
                "confidence": 0.0,
            }

        system_prompt = self.visual_prompt_manager.build_system_prompt()
        user_prompt = self.visual_prompt_manager.build_prompt(triple["head"])

        response = self._call_vlm(image_path, user_prompt, system_prompt)
        if not response:
            return {
                "reasoning": "Empty VLM response",
                "verdict": "uncertain",
                "confidence": 0.0,
            }

        result = self._safe_parse_json(response)
        if result:
            raw_verdict = result.get("verdict", "uncertain")
            if isinstance(raw_verdict, bool):
                verdict = "true" if raw_verdict else "false"
            elif isinstance(raw_verdict, str):
                verdict = raw_verdict.lower().strip()
                if verdict not in ["true", "false", "uncertain"]:
                    verdict = "uncertain"
            else:
                verdict = "uncertain"

            return {
                "reasoning": result.get("reasoning", ""),
                "verdict": verdict,
                "confidence": result.get("confidence", 0.5),
            }

        response_upper = response.upper()
        if "UNCERTAIN" in response_upper:
            verdict = "uncertain"
        elif "TRUE" in response_upper and "FALSE" not in response_upper:
            verdict = "true"
        elif "FALSE" in response_upper:
            verdict = "false"
        else:
            verdict = "uncertain"

        return {
            "reasoning": response,
            "verdict": verdict,
            "confidence": 0.5,
        }

    # ============================================================
    # Core Evaluation
    # ============================================================

    def evaluate_chunk(
        self,
        raw_chunk: str,
        triples: List[str],
        vis_triples: List[str],
        img_dict: Dict[str, str],
    ) -> Dict[str, Any]:
        verified_count = 0
        hallucination_count = 0

        for triple_str in triples:
            parsed = self._parse_triple(triple_str)
            if not parsed:
                continue

            result = self._verify_text_triple(parsed, raw_chunk)
            if result.get("verdict"):
                verified_count += 1
            else:
                hallucination_count += 1

        for triple_str in vis_triples:
            parsed = self._parse_visual_triple(triple_str)
            if not parsed:
                continue

            image_path = img_dict.get(parsed["tail"], "")
            if not image_path:
                self.logger.warning(
                    f"Image ID {parsed['tail']} not found in img_dict for triple: {triple_str}"
                )
                continue

            result = self._verify_visual_triple(parsed, image_path)
            verdict = result.get("verdict", "uncertain")
            if verdict == "true" or verdict is True:
                verified_count += 1
            elif verdict == "false" or verdict is False:
                hallucination_count += 1

        return {
            "hallucination_rate": round(
                self._calculate_hallucination_rate(
                    verified=verified_count,
                    hallucinations=hallucination_count,
                ),
                4,
            )
        }

    def process_batch(self, records: List[Dict[str, Any]]):
        results = []

        for row in tqdm(records, desc="Hallucination Eval"):
            text = row.get("raw_chunk", "")
            triples = self._ensure_list(row.get("triple", []))
            vis_triples = self._ensure_list(row.get("vis_triple", []))
            img_dict = self._ensure_dict(row.get("img_dict", {}))

            if not text:
                results.append(self._empty_result())
                continue

            try:
                results.append(
                    self.evaluate_chunk(
                        raw_chunk=text,
                        triples=triples,
                        vis_triples=vis_triples,
                        img_dict=img_dict,
                    )
                )
            except Exception as e:
                self.logger.error(f"LLM Error: {e}", exc_info=True)
                results.append(self._empty_result())

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "raw_chunk",
        input_key_meta1: str = "triple",
        input_key_meta2: str = "vis_triple",
        input_key_meta3: str = "img_dict",
        output_key: str = "hallucination_rate",
    ):
        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []
        for _, r in df.iterrows():
            records.append(
                {
                    "raw_chunk": r.get(input_key, ""),
                    "triple": r.get(input_key_meta1, []),
                    "vis_triple": r.get(input_key_meta2, []),
                    "img_dict": r.get(input_key_meta3, {}),
                }
            )

        outputs = self.process_batch(records)

        df[output_key] = [
            o.get("hallucination_rate", 0.0)
            for o in outputs
        ]

        out_file = storage.write(df)

        avg_hallucination_rate = (
            df[output_key].mean()
            if len(df) > 0 else 0.0
        )

        self.logger.info(
            f"Avg Hallucination Rate: {avg_hallucination_rate:.4f}. "
            f"Saved to {out_file}"
        )

        return [output_key]