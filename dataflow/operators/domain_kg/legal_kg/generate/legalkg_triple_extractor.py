from dataflow.prompts.diverse_kg.legalkg import LegalKGRelationExtractorPrompt
from dataflow.prompts.diverse_kg.legalkg import LegalKGAttributeExtractorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from typing import Any, Dict, List, Optional
import json
from tqdm import tqdm
import re

from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from typing import Union


@prompt_restrict(
    LegalKGRelationExtractorPrompt,
    LegalKGAttributeExtractorPrompt
)
@OPERATOR_REGISTRY.register()
class LegalKGTupleExtraction(OperatorABC):
    r"""
    A processor for extracting knowledge graph triples from text.

    This operator takes raw text and a predefined list of valid entities as input,
    and uses an LLM-based prompt to extract entity–relation–entity triples.
    The extracted triples are written back to the dataframe for downstream
    knowledge graph construction or reasoning tasks.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        triple_type: str = "attribute",
        lang: str = "zh",
        num_q: int = 5
    ):
        """
        Initialize the KGTripleExtraction operator.

        Args:
            llm_serving: LLM serving backend used for prompt inference.
            seed: Random seed for reproducibility.
            lang: Language setting for the prompt.
            prompt_template: Optional custom prompt template.
            num_q: Reserved parameter for future extensions.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if triple_type == "attribute":
            self.prompt_template = (
                LegalKGAttributeExtractorPrompt(lang=self.lang)
            )
        elif triple_type == "relation":
            self.prompt_template = (
                LegalKGRelationExtractorPrompt(lang=self.lang)
            )            

    @staticmethod
    def get_desc(lang: str = "en") :
        """
        Return a short description of the operator.

        Args:
            lang: Language of the description.

        Returns:
            A triple containing a brief description and the expected input/output.
        """
        if lang == "zh":
            return (
                "GeoKGTripleExtraction 是一个三元组抽取算子，用于从文本中抽取知识图谱三元组。",
                "输入为原始文本及其对应的合法实体列表，输出为结构化的三元组结果。"
            )
        else:
            return (
                "GeoKGTripleExtraction extracts triples from text.",
                "Input: raw text and a list of valid entities. Output: extracted KG triples."
            )

    def process_batch(
        self,
        texts: List[str],
        ontology_lists,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of texts for triple extraction.

        Args:
            texts: List of input text chunks.
            entity_lists: List of valid entity lists aligned with texts.
            sources: Optional source identifiers.

        Returns:
            A list of extraction results.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [
            {
                "text": text,
                "source": source
            }
            for text, source in zip(texts, sources)
        ]

        return self._construct_examples(raw_data, ontology_lists)

    # def _validate_dataframe(self, dataframe: pd.DataFrame):
    #     required_keys = [self.input_key]
    #     forbidden_keys = [self.output_key]

    #     missing = [k for k in required_keys if k not in dataframe.columns]
    #     conflict = [k for k in forbidden_keys if k in dataframe.columns]

    #     if missing:
    #         raise ValueError(f"Missing required column(s): {missing}")
    #     if conflict:
    #         raise ValueError(
    #             f"The following column(s) already exist and would be overwritten: {conflict}"
    #         )

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists = None,
        input_key: str = "raw_chunk",
        input_key_meta: str = "legal_ontology",
        output_key: str = "triple",
        output_key_meta1: str = "entity_class",
        output_key_meta2: str = "case_summary",
    ):
        self.input_key = input_key
        self.input_key_meta = input_key_meta
        self.output_key = output_key
        self.output_key_meta1 = output_key_meta1
        self.output_key_meta2 = output_key_meta2

        dataframe = storage.read("dataframe")
        # self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()

        if ontology_lists == None:
            storage_meta = FileStorage(
                first_entry_file_name="",  # 可以留空
                cache_type="json"
            )    
            ontology_lists = storage_meta.read(file_path=f"./.cache/api/{input_key_meta}.json", output_type="dataframe")
            row = ontology_lists.iloc[0]
            ontology_dict = {
                "entity_type": row["entity_type"],
                "relation_type": row["relation_type"],
                "attribute_type": row.get("attribute_type", {})  # 可选
            }

        outputs = self.process_batch(texts, ontology_dict)

        dataframe[self.output_key] = [
            o.get(self.output_key, []) for o in outputs
        ]

        dataframe[self.output_key_meta1] = [
            o.get(self.output_key_meta1, []) for o in outputs
        ]

        dataframe[self.output_key_meta2] = [
            o.get(self.output_key_meta2, []) for o in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    # ------------------------------------------------------------------
    # Internal helper functions (formerly ExampleConstructor)
    # ------------------------------------------------------------------

    def _construct_examples(
        self, raw_data: List[Dict[str, Any]], ontology_lists
    ) -> List[Dict[str, Any]]:
        """
        Construct extraction results from raw inputs.
        """
        self.logger.info("Starting triple extraction...")
        results = []

        for data in tqdm(raw_data, desc="Extract triples"):
            processed_text = self._preprocess_text(data.get("text", ""))
            if not processed_text:
                results.append(
                    {
                        "source_text": "",
                        "triple": [],
                    }
                )
                continue


            user_inputs = [
                self.prompt_template.build_prompt(processed_text)
            ]
            system_prompt = self.prompt_template.build_system_prompt(ontology_lists)

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            triples = self._triple_parse_llm_response(responses[0])
            entity_class = self._class_parse_llm_response(responses[0])
            case_summary = self._summary_parse_llm_response(responses[0])

            results.append(
                {
                    "source_text": processed_text,
                    "triple": triples,
                    "entity_class": entity_class,
                    "case_summary": case_summary
                }
            )

        return results

    def _triple_parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("triple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _class_parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("entity_class", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []


    def _summary_parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("case_summary", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and validate input text before extraction.
        """
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
        """
        Basic text quality checks.
        """
        if text.count("。") < 2 and text.count(".") < 2:
            return False

        if self._calculate_special_char_ratio(text) > 0.3:
            return False

        return True
