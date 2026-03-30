from dataflow.prompts.diverse_kg.medkg import MedKGRelationExtractorPrompt
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
    MedKGRelationExtractorPrompt,
)
@OPERATOR_REGISTRY.register()
class MedKGTripleExtraction(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        triple_type: str = "relation",
        lang: str = "en",
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
        self.prompt_template = (
                MedKGRelationExtractorPrompt(lang=self.lang)
            )        

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGTripleExtraction 是一个三元组抽取算子，用于从文本中抽取知识图谱三元组。",
                "输入为原始文本及其对应的合法实体列表，输出为结构化的三元组结果。"
            )
        else:
            return (
                "MedKGTripleExtraction extracts triples from text.",
                "Input: raw text and a list of valid entities. Output: extracted KG triples."
            )

    def process_batch(
        self,
        texts: List[str],
        ontology_lists: Dict[str, Any],
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

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key, self.output_key_meta]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def _load_single_ontology_from_cache(
        self,
        storage_meta: FileStorage,
        ontology_name: str,
    ) -> Dict[str, Any]:
        ontology_file_name = ontology_name if ontology_name.endswith(".json") else f"{ontology_name}.json"
        ontology_data = storage_meta.read(
            file_path=f"./.cache/medical/{ontology_file_name}",
            output_type="dataframe"
        )
        return self._normalize_ontology(ontology_data)

    def _normalize_ontology(self, ontology_data: Any) -> Dict[str, Any]:
        if isinstance(ontology_data, pd.DataFrame):
            if ontology_data.empty:
                raise ValueError("Ontology dataframe is empty")
            ontology_data = ontology_data.iloc[0].to_dict()

        if not isinstance(ontology_data, dict):
            raise ValueError("Ontology must be a dict or a dataframe containing one ontology row")

        return ontology_data

    def _merge_ontology_sections(
        self,
        merged_section: Dict[str, List[str]],
        section_data: Any,
    ) -> None:
        if not isinstance(section_data, dict):
            return

        for group_name, values in section_data.items():
            if not isinstance(values, list):
                continue

            merged_values = merged_section.setdefault(group_name, [])
            existing = set(merged_values)

            for value in values:
                value = str(value)
                if value not in existing:
                    merged_values.append(value)
                    existing.add(value)

    def _merge_ontologies(self, ontology_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not ontology_list:
            raise ValueError("Ontology list must not be empty")

        merged_ontology = {
            "entity_type": {},
            "relation_type": {},
        }

        for ontology in ontology_list:
            normalized_ontology = self._normalize_ontology(ontology)
            self._merge_ontology_sections(
                merged_ontology["entity_type"],
                normalized_ontology.get("entity_type", {}),
            )
            self._merge_ontology_sections(
                merged_ontology["relation_type"],
                normalized_ontology.get("relation_type", {}),
            )

        return merged_ontology

    def _resolve_ontology(
        self,
        ontology_lists: Optional[Any],
        input_key_meta: Union[str, List[str]],
    ) -> Dict[str, Any]:
        if ontology_lists is not None:
            if isinstance(ontology_lists, list):
                return self._merge_ontologies(ontology_lists)
            return self._normalize_ontology(ontology_lists)

        storage_meta = FileStorage(
            first_entry_file_name="",
            cache_type="json"
        )

        if isinstance(input_key_meta, list):
            loaded_ontologies = [
                self._load_single_ontology_from_cache(storage_meta, ontology_name)
                for ontology_name in input_key_meta
            ]
            return self._merge_ontologies(loaded_ontologies)

        return self._load_single_ontology_from_cache(storage_meta, input_key_meta)

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists = None,
        input_key: str = "raw_chunk",
        input_key_meta: Union[str, List[str]] = "ontology",
        output_key: str = "triple",
        output_key_meta: str = "entity_class"
    ):
        self.input_key = input_key
        self.input_key_meta = input_key_meta
        self.output_key = output_key
        self.output_key_meta = output_key_meta

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        ontology_lists = self._resolve_ontology(ontology_lists, input_key_meta)

        outputs = self.process_batch(texts, ontology_lists)

        dataframe[self.output_key] = [
            o.get(self.output_key, []) for o in outputs
        ]
        dataframe[self.output_key_meta] = [
            o.get(self.output_key_meta, []) for o in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key, output_key_meta]

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
                        "entity_class": [],
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

            triples = self._parse_llm_response(responses[0])
            entity_class = self._class_parse_llm_response(responses[0])

            results.append(
                {
                    "source_text": processed_text,
                    "triple": triples,
                    "entity_class": entity_class,
                }
            )

        return results

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
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
