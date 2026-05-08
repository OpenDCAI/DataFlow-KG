import pandas as pd
import random
import re
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


class KGTupleRemoveRepeated(OperatorABC):
    """
    Operator for cleaning and deduplicating knowledge graph triples or tuples.

    Supports:
    1. Relation triples:
       "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by"
    2. Attribute triples:
       "<entity> Henry <attribute> profession <value> musician"
    3. Any n-tuples with <tag> markers.

    Deduplication is STRICT: only completely identical strings are removed.
    The original nested structure is strictly preserved.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        seed: int = 0,
        lang: str = "en"
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTupleRemoveRepeated 用于清洗和去重关系三元组、属性三元组或多元组。",
                "仅删除完全相同的元组字符串，保留原始嵌套结构。",
                "输入列 triple（或 tuple）为元组字符串的嵌套列表，输出列同名，包含去重后的元组列表。"
            )
        else:
            return (
                "KGTupleRemoveRepeated cleans and deduplicates relation, attribute, or general n-tuples.",
                "Only fully identical tuple strings are removed; the original nested structure is preserved.",
                "Takes triple (or tuple) as input column and outputs the same column with duplicate tuples removed."
            )

    def _remove_duplicates_strict(
        self, nested_list: List[List[str]]
    ) -> List[List[str]]:
        """
        Strict deduplication across all rows:
        - Merge all sublists into a single list
        - Remove exactly identical strings
        - Return a single cleaned list wrapped in a 2D list
        """
        # Flatten all triples
        all_items = [item for sublist in nested_list for item in sublist]

        # Deduplicate strictly
        seen = set()
        cleaned_all = []
        for item in all_items:
            if item not in seen:
                seen.add(item)
                cleaned_all.append(item)

        # Wrap in a single sublist to preserve 2D structure
        return [cleaned_all]

    # ========== DataFrame validation ==========
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        # 自动选择输入列
        if hasattr(self, "input_key") and self.input_key in dataframe.columns:
            chosen_input_key = self.input_key
        elif "triple" in dataframe.columns:
            chosen_input_key = "triple"
        elif "tuple" in dataframe.columns:
            chosen_input_key = "tuple"
        else:
            raise ValueError(
                "Missing required input column: neither 'triple' nor 'tuple' found in dataframe"
            )
        self.input_key = chosen_input_key

        # 设置默认输出列
        self.output_key = "triple" if self.input_key == "triple" else "tuple"

        self.logger.info(f"Using input column '{self.input_key}' and output column '{self.output_key}'")


    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        output_key: str = "triple",
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        nested_list = dataframe[self.input_key].tolist()
        cleaned = self._remove_duplicates_strict(nested_list)

        df = pd.DataFrame()
        df[self.output_key] = cleaned
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]