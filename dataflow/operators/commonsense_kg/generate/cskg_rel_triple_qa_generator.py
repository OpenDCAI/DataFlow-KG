from dataflow.prompts.diverse_kg.cskg import (
  CSKGSingleRelationTripleQAPrompt,
  CSKGSetBasedTripleQAPrompt,
  CSKGMultiRelationTripleQAPrompt
)

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict

import random
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import json
import re


@prompt_restrict(
    CSKGSingleRelationTripleQAPrompt,
    CSKGSetBasedTripleQAPrompt,
    CSKGMultiRelationTripleQAPrompt
)

@OPERATOR_REGISTRY.register()
class CSKGRelationTripleQAGeneration(OperatorABC):
    """
    KGAttributeTripleQAGeneration generates attribute-based QA pairs.

    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        qa_type: str = "set",
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        self.qa_type = qa_type

        if self.qa_type == "single":
            self.prompt_template = CSKGSingleRelationTripleQAPrompt(lang=self.lang)
        elif self.qa_type == "set":
            self.prompt_template = CSKGSetBasedTripleQAPrompt(lang=self.lang)
        elif self.qa_type == 'multi':
            self.prompt_template = CSKGMultiRelationTripleQAPrompt(lang=self.lang)
        else:
            raise ValueError(f"Unsupported triple_type: {self.qa_type}")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "CSKGTripleQAGeneration 用于基于实体属性信息生成问答对。"
            )
        else:
            return (
                "KGAttributeTripleMultiEntityQAGeneration generates QA pairs based on attributes of entities."
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

        for triples, source in tqdm(
            zip(texts, sources),
            total=len(texts),
            desc="Generating QA pairs",
        ):
            user_prompt = self.prompt_template.build_prompt(triples)
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )

            qa_pairs = self._parse_llm_response(responses[0])
            results.append({"QA_pairs": qa_pairs})

        return results

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "QA_pairs",
    ):
        self.input_key = input_key
        if self.qa_type == 'set':
            self.input_key = 'set_triple'

        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [o[self.output_key] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column already exists: {self.output_key}")

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse QA_pairs from the LLM response.
        The response is expected to be a JSON object containing a 'QA_pairs' field.
        """
        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
            return json.loads(json_str).get("QA_pairs", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []
