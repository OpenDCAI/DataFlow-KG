from dataflow.prompts.diverse_kg.geokg import GeoKGEventExtractorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import random
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import json
import re


@prompt_restrict(GeoKGEventExtractorPrompt)
@OPERATOR_REGISTRY.register()
class GeoKGEventExtraction(OperatorABC):
    """
    KGEntityExtraction identifies entity mentions from raw text.

    The operator:
    - Takes raw text as input
    - Uses an LLM to extract entity surface forms
    - Outputs a normalized entity list for downstream KG construction
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[GeoKGEventExtractorPrompt, DIYPromptABC] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.prompt_template = (
            prompt_template
            if prompt_template
            else GeoKGEventExtractorPrompt(lang=self.lang)
        )

        self.min_text_length = 10
        self.max_text_length = 200000

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGEntityExtraction 用于从原始文本中识别并抽取实体。",
                "该算子仅执行实体识别，不涉及实体关系或属性抽取。",
                "通常作为知识图谱构建流程中的实体候选生成阶段。"
            )
        else:
            return (
                "KGEntityExtraction extracts entity mentions from raw text.",
                "The operator performs entity identification only, without relation or attribute extraction.",
                "It is typically used as the entity candidate generation stage in KG pipelines."
            )

    def process_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        results = []

        for text, source in tqdm(
            zip(texts, sources),
            total=len(texts),
            desc="Extracting entities",
        ):
            processed_text = self._preprocess_text(text)
            if not processed_text:
                results.append({"tuple": ""})
                continue

            user_prompt = self.prompt_template.build_prompt(processed_text)
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )

            entities = self._parse_llm_response(responses[0])
            results.append({"tuple": entities})

        return results

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "raw_chunk",
        output_key: str = "tuple",
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [o["tuple"] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column already exists: {self.output_key}")

    def _parse_llm_response(self, response: str) -> str:
        """
        Parse entity list from the LLM response.
        The response is expected to be a JSON array of entity strings.
        """
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("tuple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.strip()
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            return ""

        if not self._check_text_quality(text):
            return ""

        return text

    def _check_text_quality(self, text: str) -> bool:
        if text.count(".") + text.count("。") < 2:
            return False
        return self._calculate_special_char_ratio(text) <= 0.3

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

        return special_count / len(text) if text else 0

    def _normalize_text_key(self, text: str) -> str:
        stopwords = {
            "the", "a", "an", "of", "and", "or", "in", "on", "at",
            "to", "for", "with", "by", "as", "from", "into",
        }

        pattern = r"\b(" + "|".join(stopwords) + r")\b"
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
