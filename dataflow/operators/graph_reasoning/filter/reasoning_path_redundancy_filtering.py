# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KGPathRedundancyFilter
====================================
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List

@OPERATOR_REGISTRY.register()
class KGReasoningPathRedundancyFilter(OperatorABC):
    """
    Filter multi-hop KG paths based on redundancy scores.

    Input DataFrame should have:
      - column `mpath` (List[List[List[str]]])
      - column `redundancy_scores` (List[List[float]] aligned with mpath)
    
    Output:
      - column `filtered_mpath` containing paths with score within [min_score, max_score]
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGReasoningPathRedundancyFilter 用于按冗余度得分筛选图谱多跳路径。",
                "输入: mpath + redundancy_scores; 输出: filtered_mpath",
            )
        return (
            "KGReasoningPathRedundancyFilter is used to filter multi-hop reasoning paths by redundancy scores.",
            "Input: mpath + redundancy_scores; Output: filtered_mpath",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, score_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if score_key not in df.columns:
            raise ValueError(f"Missing required column: {score_key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "mpath",
        score_key: str = "redundancy_scores",
        output_key: str = "filtered_mpath",
        min_score: float = 0.0,
        max_score: float = 0.5,
    ):
        """
        Filter multi-hop paths based on redundancy score.

        Args:
            input_key: column name for paths (mpath)
            score_key: column name for redundancy scores
            output_key: column name for filtered paths
            min_score: minimum allowed redundancy score
            max_score: maximum allowed redundancy score
        """
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key, score_key)
        self.logger.info(f"Filtering paths with redundancy score in [{min_score}, {max_score}]")

        filtered_results = []

        for paths_list, scores_list in zip(df[input_key], df[score_key]):
            if not isinstance(paths_list, list) or not isinstance(scores_list, list):
                filtered_results.append([])
                continue

            # 对每条路径及其分数进行筛选
            filtered_paths = [
                p for p, s in zip(paths_list, scores_list)
                if s is not None and min_score <= s <= max_score
            ]
            filtered_results.append(filtered_paths)

        if self.merge_to_input:
            df[input_key] = filtered_results
            output_file = storage.write(df)
            self.logger.info(f"Results saved to {output_file}")
            return [input_key]

        df[output_key] = filtered_results
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]
