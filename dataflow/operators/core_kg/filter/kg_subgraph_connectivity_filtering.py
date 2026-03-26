import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List, Optional


@OPERATOR_REGISTRY.register()
class KGSubgraphConnectivityFilter(OperatorABC):
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
                "KGSubgraphConnectivityFilter 根据连通性得分筛选子图。",
                "读取子图数据及对应的连通性得分列，保留得分在指定范围内的行。",
                "输入列 subgraph 为子图三元组列表，density（可配置）为连通性得分列；过滤后直接写回 DataFrame，不新增输出列。",
            )
        return (
            "KGSubgraphConnectivityFilter filters subgraphs based on connectivity scores.",
            "Reads subgraph data and their connectivity score column, retaining only rows within the specified score range.",
            "Takes subgraph (List[str]) and a score column (default: density) as inputs; filtered rows are written back to the DataFrame without adding new columns.",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, output_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if output_key not in df.columns:
            raise ValueError(f"Missing required column: {output_key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "subgraph",
        output_key: str = "density",
        min_score: float = 0.3,
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