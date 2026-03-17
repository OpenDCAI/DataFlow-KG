# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KGSubgraphConsistency
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-03-14
License:
    MIT License
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from typing import Any, Dict, List, Optional, Union
import json
import re
from tqdm import tqdm
import random

from dataflow.prompts.diverse_kg.legalkg import CaseSummarySimilarityPrompt


class LegalKGCaseSummarySimilarity(OperatorABC):
    """
    使用 LLM 评估案件摘要(case_summary)与案件类型(case_type)描述的语义相似度。
    输出 similarity_score（0~1）。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "zh",
        merge_to_input: bool = False,
        prompt_template: Union[CaseSummarySimilarityPrompt, DIYPromptABC] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang.lower()
        self.logger = get_logger()
        self.merge_to_input = merge_to_input
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else CaseSummarySimilarityPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "CaseSummarySimilarity 用于评估案件摘要与案件类型描述的语义匹配度。",
                "输入 case_summary 和 case_type，输出 similarity_score（0~1）。"
            )
        else:
            return (
                "CaseSummarySimilarity evaluates semantic similarity between a case summary and a case type description.",
                "Input: case_summary and case_type; Output: similarity_score (0-1)."
            )

    def process_batch(
        self,
        summaries: List[str],
        case_types: List[str],
    ) -> List[Dict[str, Any]]:
        results = []
        for summary, case_type in tqdm(zip(summaries, case_types), total=len(summaries), desc="Scoring similarity"):
            user_input = [self.prompt_template.build_prompt(summary, case_type)]
            system_prompt = self.prompt_template.build_system_prompt()

            response = self.llm_serving.generate_from_input(
                user_inputs=user_input,
                system_prompt=system_prompt,
            )

            try:
                parsed_output = json.loads(
                    re.sub(r"```json|```|\n", "", response[0])
                )
                score = parsed_output.get("similarity_score")
            except Exception as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                score = None

            results.append({"similarity_score": score})

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str):
        for key in [input_key]:
            if key not in dataframe.columns:
                raise ValueError(f"Missing required column: {key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "case_summary",
        input_key_meta: str = "盗窃案件",
        output_key: str = "similarity_score",
    ):
        """
        Run similarity evaluation on DataFrame stored in DataFlowStorage.
        """
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe, input_key)

        summaries = dataframe[input_key].tolist()

        outputs = self.process_batch(summaries, input_key_meta)

        dataframe[output_key] = [o["similarity_score"] for o in outputs]
        storage.write(dataframe)
        return [output_key]