import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List, Optional


@OPERATOR_REGISTRY.register()
class LegalKGCaseSimilarityFilter(OperatorABC):
    """
    Filter knowledge graph subgraphs based on consistency scores.

    Input DataFrame should have:
      - column `subgraph` (List[str])
      - column `consistency_score` (float)
    
    Output:
      - column `filtered_subgraph` containing subgraphs with score within [min_score, max_score]
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGSubgraphConsistenceFilter 根据子图一致性得分筛选子图。",
                "输入列需包含子图及其一致性得分，输出符合得分范围的子图。",
                "输入: subgraph, consistency_score\n输出: filtered_subgraph",
            )
        return (
            "KGSubgraphConsistenceFilter filters knowledge graph subgraphs based on consistency scores.",
            "Input columns must contain subgraphs and their consistency scores.",
            "Output column: filtered_subgraph",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, output_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if output_key not in df.columns:
            raise ValueError(f"Missing required column: {output_key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        output_key: str = "similarity_score",
        min_score: float = 0.8,
        max_score: float = 1.0,
    ):
        """
        Filter subgraphs based on consistency score.
        Rows not meeting the score threshold are removed.

        Args:
            input_key: column name for subgraph list
            output_key: column name for consistency score
            output_key: column name for filtered subgraphs
            min_score: minimum allowed score
            max_score: maximum allowed score
        """
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key, output_key)
        self.logger.info(f"Filtering subgraphs with consistency_score in [{min_score}, {max_score}]")

        # 保留符合要求的行
        filtered_df = df[(df[output_key] >= min_score) & (df[output_key] <= max_score)]

        output_file = storage.write(filtered_df)
        self.logger.info(f"Results saved to {output_file}")

        return None