# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KGMultiHopPathFilterByLength
====================================
"""

from typing import List
from tqdm import tqdm
import pandas as pd

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KGReasoningPathLengthFilter(OperatorABC):
    """
    Filter multi-hop KG paths by their length.

    Input columns:
        - mpath: List[List[List[str]]]   # 原始路径
        - mpath_length: List[List[int]]  # 每条路径的长度

    Output columns:
        - filtered_mpath: 同结构，只保留长度在指定范围的路径
    """

    def __init__(self, min_length: int = 1, max_length: int = 10):
        super().__init__()
        self.logger = get_logger()
        self.min_length = min_length
        self.max_length = max_length

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGReasoningPathLengthFilter 用于按长度筛选图谱多跳路径。",
                "输入: mpath + mpath_length; 输出: filtered_mpath",
            )
        return (
            "KGReasoningPathLengthFilter is used to filter multi-hop reasoning paths by length.",
            "Input: mpath + mpath_length; Output: filtered_mpath",
        )

    # ----------------------------
    # Validation
    # ----------------------------
    def _validate_dataframe(self, df: pd.DataFrame):
        for col in ["mpath", "mpath_length"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

    # ----------------------------
    # Run
    # ----------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        mpath_key: str = "mpath",
        length_key: str = "mpath_length",
        output_path_key: str = "filtered_mpath",
    ) -> List[str]:
        """
        Filter paths based on length range [min_length, max_length].
        """

        if storage is None:
            raise ValueError("storage cannot be None")

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        filtered_paths_all = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering paths by length"):
            row_paths = row.get(mpath_key, [])
            row_lengths = row.get(length_key, [])

            if not isinstance(row_paths, list) or not isinstance(row_lengths, list):
                filtered_paths_all.append([])
                continue

            row_filtered_paths = []

            for pair_idx, pair_paths in enumerate(row_paths):
                pair_lengths = row_lengths[pair_idx] if pair_idx < len(row_lengths) else []

                if not isinstance(pair_paths, list) or not isinstance(pair_lengths, list):
                    row_filtered_paths.append([])
                    continue

                # 根据长度筛选
                filtered_pair_paths = [
                    path for path, length in zip(pair_paths, pair_lengths)
                    if self.min_length <= length <= self.max_length
                ]

                row_filtered_paths.append(filtered_pair_paths)

            filtered_paths_all.append(row_filtered_paths)

        df[output_path_key] = filtered_paths_all
        out_file = storage.write(df)
        self.logger.info(f"Filtered paths saved to {out_file}")

        return [output_path_key]
