# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: FinKGTupleExtraction
====================================

License:
    MIT License
"""

from dataflow.prompts.diverse_kg.finkg import FinKGRelationExtractorPrompt
from dataflow.prompts.diverse_kg.finkg import FinKGAttributeExtractorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from typing import Any, Dict, List, Optional
import json
from tqdm import tqdm

from dataflow.core.prompt import prompt_restrict


@prompt_restrict(
    FinKGRelationExtractorPrompt,
    FinKGAttributeExtractorPrompt
)
@OPERATOR_REGISTRY.register()
class FinKGTupleExtraction(OperatorABC):
    r"""
    Extract financial knowledge graph quadruples from text using LLM.

    Supports two quadruple types:
      - relation: <subj> Entity <obj> Entity <rel> Relation <time> TimeValue
      - attribute: <subj> Entity <attribute> Attribute <value> AttributeValue <time> TimeValue

    Each quadruple is accompanied by entity_class labels from the ontology.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        triple_type: str = "relation",
        lang: str = "en",
        num_q: int = 5
    ):
        _ = seed, num_q
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        if triple_type == "attribute":
            self.prompt_template = (
                FinKGAttributeExtractorPrompt(lang=self.lang)
            )
        elif triple_type == "relation":
            self.prompt_template = (
                FinKGRelationExtractorPrompt(lang=self.lang)
            )
        else:
            raise ValueError(
                f"Invalid triple_type '{triple_type}'. Must be 'relation' or 'attribute'."
            )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGTupleExtraction 用于从金融文本中抽取知识图谱四元组。",
                "输入: 原始文本 + 本体; 输出: 关系/属性四元组 + 实体类别",
            )
        return (
            "FinKGTupleExtraction is used to extract knowledge graph quadruples from financial text.",
            "Input: raw text + ontology; Output: relation/attribute quadruples + entity classes.",
        )

    def process_batch(
        self,
        texts: List[str],
        ontology: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return self._construct_examples(texts, ontology)

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists=None,
        input_key: str = "raw_chunk",
        input_key_meta: str = "finkg_ontology",
        output_key: str = "tuple",
        output_key_meta: str = "entity_class"
    ):
        dataframe = storage.read("dataframe")

        texts = dataframe[input_key].tolist()
        ontology = load_finkg_ontology(
            ontology_lists=ontology_lists,
            input_key_meta=input_key_meta,
        )

        outputs = self.process_batch(texts, ontology)

        dataframe[output_key] = [
            o.get("tuple", []) for o in outputs
        ]

        dataframe[output_key_meta] = [
            o.get("entity_class", []) for o in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    # ------------------------------------------------------------------
    # Internal helper functions
    # ------------------------------------------------------------------

    def _construct_examples(
        self, texts: List[str], ontology: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        self.logger.info("Starting quadruple extraction...")
        results = []

        for text in tqdm(texts, desc="Extract quadruples"):
            processed_text = self._preprocess_text(text)
            if not processed_text:
                results.append(
                    {
                        "source_text": "",
                        "tuple": [],
                        "entity_class": []
                    }
                )
                continue

            user_inputs = [
                self.prompt_template.build_prompt(processed_text)
            ]
            system_prompt = self.prompt_template.build_system_prompt(ontology)

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            tuples = self._tuple_parse_llm_response(responses[0])
            entity_class = self._class_parse_llm_response(responses[0])

            results.append(
                {
                    "source_text": processed_text,
                    "tuple": tuples,
                    "entity_class": entity_class
                }
            )

        return results

    def _tuple_parse_llm_response(self, response: str) -> List[str]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("tuple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _class_parse_llm_response(self, response: str) -> List[List[str]]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("entity_class", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.strip()

        if len(text) < 10 or len(text) > 200000:
            return ""

        if not self._check_text_quality(text):
            return ""

        return text

    def _calculate_special_char_ratio(self, text: str) -> float:
        chinese_ranges = [
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
        ]

        special_count = 0
        for c in text:
            is_chinese = any(start <= ord(c) <= end for start, end in chinese_ranges)
            if not (c.isalnum() or c.isspace() or is_chinese):
                special_count += 1

        return special_count / len(text) if text else 0.0

    def _check_text_quality(self, text: str) -> bool:
        if text.count("。") < 2 and text.count(".") < 2:
            return False

        if self._calculate_special_char_ratio(text) > 0.3:
            return False

        return True
