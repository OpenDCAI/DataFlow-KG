"""
====================================
DataFlow-KG: 
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

from dataflow.prompts.core_kg.rel_triple_filter import KGEntityValidityPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import random
from typing import Any, Dict, List, Optional, Union
import json
import re
from tqdm import tqdm


@prompt_restrict(KGEntityValidityPrompt)
@OPERATOR_REGISTRY.register()
class KGEntityValidity(OperatorABC):
    """
    KGEntityValidity validates extracted entities for knowledge graph construction.

    The operator uses a large language model to determine whether entities
    are semantically valid and suitable for inclusion in a knowledge graph.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        merge_to_input = False,
        prompt_template: Union[KGEntityValidityPrompt, DIYPromptABC] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else KGEntityValidityPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """
        Return a human-readable description of this operator.
        """
        if lang == "zh":
            return (
                "KGEntityValidity 用于判断知识图谱候选实体的有效性。",
                "该算子使用大语言模型评估实体在语义上是否合理、是否适合作为知识图谱节点。",
                "输入为候选实体，输出为实体有效性判断结果。",
            )
        else:
            return (
                "KGEntityValidity validates candidate entities for knowledge graphs.",
                "It uses a large language model to assess whether entities are semantically valid and meaningful.",
                "Input consists of candidate entities, and output provides validity judgments.",
            )

    def process_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate entities in batch using the LLM.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        results = []

        for text, source in tqdm(
            zip(texts, sources),
            total=len(texts),
            desc="Validating entities",
        ):
            user_inputs = [self.prompt_template.build_prompt(text)]
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            parsed_output = json.loads(
                re.sub(r"```json|```|\n", "", responses[0])
            )

            results.append(
                {
                    "source_text": text,
                    self.output_key: parsed_output,
                }
            )

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate required input and output columns.
        """
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(
                f"Column '{self.output_key}' already exists and would be overwritten"
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "entity",
        output_key: str = "valid",
    ):
        """
        Run entity validity checking on a DataFlow dataframe.
        """
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        if self.merge_to_input:
            dataframe[self.input_key] = [o[self.output_key] for o in outputs]
            storage.write(dataframe)
            return [input_key]
        else:
            dataframe[self.output_key] = [o[self.output_key] for o in outputs]
            storage.write(dataframe)
            return [output_key]
