from dataflow.prompts.diverse_kg.cskg import CSKGRelationtripleExtractorPrompt
from dataflow.prompts.diverse_kg.cskg import CSKGAttributetripleExtractorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
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
    CSKGRelationtripleExtractorPrompt,
    CSKGAttributetripleExtractorPrompt
)
@OPERATOR_REGISTRY.register()
class CSKGTripleExtraction(OperatorABC):
    r"""
    A processor for extracting commonsense knowledge graph (CSKG) triples from text.

    This operator takes raw text chunks as input and uses an LLM-based prompt 
    to extract structured triples (either relations or attributes, depending on the configuration).
    The extracted triples are written back to the dataframe for downstream
    knowledge graph construction or reasoning tasks.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        triple_type: str = "relation",
        lang: str = "en",
        num_q: int = 5
    ):
        """
        Initialize the CSKGTripleExtraction operator.

        Args:
            llm_serving: LLM serving backend used for prompt inference.
            seed: Random seed for reproducibility.
            triple_type: The type of triples to extract ("relation" or "attribute").
            lang: Language setting for the prompt.
            num_q: Reserved parameter for future extensions.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if triple_type == "attribute":
            self.prompt_template = (
                CSKGAttributetripleExtractorPrompt(lang=self.lang)
            )
        elif triple_type == "relation":
            self.prompt_template = (
                CSKGRelationtripleExtractorPrompt(lang=self.lang)
            )            

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """
        Return a short description of the operator.

        Args:
            lang: Language of the description.

        Returns:
            A tuple containing a brief description and the expected input/output.
        """
        if lang == "zh":
            return (
                "CSKGTripleExtraction 是一个常识知识图谱三元组抽取算子，支持从文本中抽取属性（attribute）或关系（relation）三元组。",
                "输入为原始文本，输出为结构化的三元组（triple）结果。"
            )
        else:
            return (
                "CSKGTripleExtraction extracts commonsense knowledge graph triples (attributes or relations) from text.",
                "Input: raw text. Output: extracted structured KG triples."
            )

    def process_batch(
        self,
        texts: List[str],
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
                "source": source,
            }
            for text, source in zip(texts, sources)
        ]

        return self._construct_examples(raw_data)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "raw_chunk",
        output_key: str = "triple"
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()

        outputs = self.process_batch(texts)


        dataframe[self.output_key] = [
            o.get("triple", []) for o in outputs 
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]

    # ------------------------------------------------------------------
    # Internal helper functions (formerly ExampleConstructor)
    # ------------------------------------------------------------------

    def _construct_examples(
        self, raw_data: List[Dict[str, Any]]
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
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            triples = self._parse_llm_response(responses[0])

            results.append(
                {
                    "source_text": processed_text,
                    "triple": triples,
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
