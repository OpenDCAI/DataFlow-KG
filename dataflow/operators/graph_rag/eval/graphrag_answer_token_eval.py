"""
====================================
DataFlow-KG: KGAnswerTokenCount
====================================
"""

import pandas as pd
from typing import List, Union
from tqdm import tqdm
import tiktoken

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KGRAGAnswerTokenCount(OperatorABC):
    """
    Count token numbers for answers.

    Input:
        answer: str | List[str]

    Output:
        answer_token_count: int | List[int]
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
    ):
        self.logger = get_logger()
        self.encoding = tiktoken.encoding_for_model(model_name)

    # --------------------------------------------------
    # Description
    # --------------------------------------------------
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRAGAnswerTokenCount 用于统计 GraphRAG 答案的 token 数量。",
                "输入: answer; 输出: answer_token_count",
            )
        else:
            return (
                "KGRAGAnswerTokenCount is used to count tokens in GraphRAG answers.",
                "Input: answer; Output: answer_token_count",
            )

    # --------------------------------------------------
    # Token count
    # --------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.answer_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.answer_key}")

        if self.output_key in dataframe.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        answer_key: str = "answer",
        output_key: str = "answer_token_count",
    ):

        self.answer_key = answer_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        token_counts = []

        for a in tqdm(
            dataframe[self.answer_key],
            total=len(dataframe),
            desc="Counting answer tokens",
        ):

            # -----------------------------
            # CASE 1: single answer
            # -----------------------------
            if isinstance(a, str):
                token_counts.append(self._count_tokens(a))

            # -----------------------------
            # CASE 2: batch answers
            # -----------------------------
            elif isinstance(a, list):
                token_counts.append(
                    [self._count_tokens(ai) for ai in a]
                )

            else:
                raise ValueError(f"Unsupported type: {type(a)}")

        dataframe[self.output_key] = token_counts

        output_file = storage.write(dataframe)
        self.logger.info(f"Answer token count saved to {output_file}")

        return [output_key]
