"""
====================================
DataFlow-KG: KGQueryEntityRelationExtraction
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

import json
import re
import random
import pandas as pd
from typing import List, Dict, Any, Union
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

from dataflow.prompts.application_kg.graph_rag import (
    KGQueryExtractionPrompt
)


@prompt_restrict(KGQueryExtractionPrompt)
@OPERATOR_REGISTRY.register()
class KGGraphRAGQueryExtraction(OperatorABC):
    """
    Extract entities and relations from user questions for KG-based RAG.
    
    Input:
        question: str
    
    Output:
        entities: List[str]
        relations: List[str]
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[
            KGQueryExtractionPrompt, DIYPromptABC
        ] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else KGQueryExtractionPrompt(lang=self.lang)
        )

    # --------------------------------------------------
    # Description
    # --------------------------------------------------
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGQueryEntityRelationExtraction：从用户问题中抽取实体与关系，用于 KG-RAG。",
                "输入列：question (str)",
                "输出列：entities (List[str]), relations (List[str])",
            )
        else:
            return (
                "KGQueryEntityRelationExtraction extracts entities and relations from user queries.",
                "Input column: question (str)",
                "Output columns: entities (List[str]), relations (List[str])",
            )

    # --------------------------------------------------
    # LLM call
    # --------------------------------------------------
    def _extract(self, question: str) -> Dict[str, List[str]]:
        user_prompt = self.prompt_template.build_prompt(question)
        system_prompt = self.prompt_template.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        cleaned = re.sub(r"```json|```|\n", "", responses[0])

        try:
            parsed = json.loads(cleaned)
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])
        except Exception as e:
            self.logger.warning(
                f"Failed to parse LLM output: {e}, raw={responses[0]}"
            )
            entities, relations = [], []

        return {
            "entities": entities,
            "relations": relations,
        }

    # --------------------------------------------------
    # DataFrame Validation
    # --------------------------------------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        for col in self.output_keys:
            if col in dataframe.columns:
                raise ValueError(f"Output column already exists: {col}")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "question",
        output_keys: List[str] = ["entities", "relations"],
    ):
        self.input_key = input_key
        self.output_keys = output_keys

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        entities_col, relations_col = [], []

        for q in tqdm(dataframe[self.input_key].tolist(), desc="Extract KG semantics"):

            # ===============================
            # CASE 1: 单条问题（str）
            # ===============================
            if isinstance(q, str):
                result = self._extract(q)
                entities_col.append(result["entities"])
                relations_col.append(result["relations"])

            # ===============================
            # CASE 2: 多问题 batch（List[str]）
            # ===============================
            elif isinstance(q, list):
                ent_list, rel_list = [], []

                for qi in q:
                    result = self._extract(qi)
                    ent_list.append(result["entities"])
                    rel_list.append(result["relations"])

                entities_col.append(ent_list)
                relations_col.append(rel_list)

            else:
                raise ValueError(f"Unsupported question type: {type(q)}")

        dataframe["entities"] = entities_col
        dataframe["relations"] = relations_col

        output_file = storage.write(dataframe)
        self.logger.info(f"Query extraction results saved to {output_file}")

        return output_keys
