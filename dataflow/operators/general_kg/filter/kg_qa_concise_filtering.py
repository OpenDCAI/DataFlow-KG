# -*- coding: utf-8 -*-
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List, Optional

@OPERATOR_REGISTRY.register()
class KGQAConciseFilter(OperatorABC):
    """
    Filter knowledge graph triples based on strength scores.

    Input DataFrame should have:
      - column `triple` (List[str])
      - column `triple_strength_score` (List[float] aligned with triples)
    
    Output:
      - column `filtered_triple` containing triples with score within [min_score, max_score]
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGQAConciseFilter 根据简洁性得分对 QA 对进行过滤。",
                "读取 QA 对及对应的简洁性得分，保留得分在指定范围内的 QA 对。",
                "输入列 QA_pairs 为 QA 问答对列表，conciseness_scores 为对应得分列表；输出列 filtered_QA_pairs 为过滤后的 QA 对列表。",
            )
        return (
            "KGQAConciseFilter filters QA pairs based on conciseness scores.",
            "Reads QA pairs and their conciseness scores, retaining only those within the specified score range.",
            "Takes QA_pairs (List of QA pairs) and conciseness_scores (List of float) as inputs, and outputs filtered_QA_pairs (List of QA pairs that pass the score threshold).",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, score_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if score_key not in df.columns:
            raise ValueError(f"Missing required column: {score_key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "QA_pairs",
        score_key: str = "conciseness_scores",
        output_key: str = "filtered_QA_pairs",
        min_score: float = 0.95,
        max_score: float = 1.0,
    ):
        """
        Filter triples based on strength score.

        Args:
            input_key: column name for triple list
            score_key: column name for triple strength score list
            output_key: column name for filtered triples
            min_score: minimum allowed score
            max_score: maximum allowed score
        """
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key, score_key)
        self.logger.info(f"Filtering triples with score in [{min_score}, {max_score}]")

        filtered_results = []
        for triple_list, score_list in zip(df[input_key], df[score_key]):
            if not isinstance(triple_list, list) or not isinstance(score_list, list):
                filtered_results.append([])
                continue
            filtered = [
                t for t, s in zip(triple_list, score_list)
                if s is not None and min_score <= s <= max_score
            ]
            filtered_results.append(filtered)

        if self.merge_to_input:
            df[input_key] = filtered_results
            output_file = storage.write(df)
            self.logger.info(f"Results saved to {output_file}")
            return [input_key]

        df[output_key] = filtered_results
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]