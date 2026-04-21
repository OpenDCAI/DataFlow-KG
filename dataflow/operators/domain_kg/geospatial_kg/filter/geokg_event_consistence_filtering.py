import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List, Optional

@OPERATOR_REGISTRY.register()
class GeoKGEventConsistenceFilter(OperatorABC):
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
                "KGTripleStrengthFilter 根据三元组强度得分筛选三元组。",
                "输入列需包含三元组及其强度得分，输出符合得分范围的三元组。",
                "输入: triple, triple_strength_score\n输出: filtered_triple",
            )
        return (
            "KGTripleStrengthFilter filters knowledge graph triples based on strength scores.",
            "Input columns must contain triples and their strength scores.",
            "Output column: filtered_triple",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, score_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if score_key not in df.columns:
            raise ValueError(f"Missing required column: {score_key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "tuple",
        score_key: str = "consistency_scores",
        output_key: str = "filtered_tuple",
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