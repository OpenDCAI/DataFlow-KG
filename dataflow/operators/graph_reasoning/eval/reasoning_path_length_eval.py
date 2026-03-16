# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KGMultiHopPathLengthEvaluator
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
class KGReasoningPathLengthEvaluator(OperatorABC):
    """
    Compute lengths of multi-hop paths in KG.

    Input columns:
        - mpath: List[List[List[str]]]
            # nested paths grouped by row -> entity pair -> paths

    Output columns:
        - mpath_length: same structure, but each path replaced by its length
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger()

    # ----------------------------
    # Validation
    # ----------------------------
    def _validate_dataframe(self, df: pd.DataFrame, input_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing input column: {input_key}")

    # ----------------------------
    # Run
    # ----------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "mpath",
        output_key: str = "mpath_length",
    ) -> List[str]:
        """
        Compute lengths of multi-hop paths from the existing 'mpath' column.
        Keeps the same nested structure: row -> entity pair -> paths -> length
        """
        if storage is None:
            raise ValueError("storage cannot be None")

        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key)

        all_lengths = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing path lengths"):
            row_mpath = row.get(input_key, [])
            if not isinstance(row_mpath, list):
                all_lengths.append([])
                continue

            row_lengths = []
            for pair_paths in row_mpath:  # 每个目标实体对
                if not isinstance(pair_paths, list):
                    row_lengths.append([])
                    continue
                lengths = [len(path) if isinstance(path, list) else 0 for path in pair_paths]
                row_lengths.append(lengths)

            all_lengths.append(row_lengths)

        df[output_key] = all_lengths
        out_file = storage.write(df)
        self.logger.info(f"Path lengths saved to {out_file}")

        return [output_key]