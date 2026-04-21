import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List, Optional
import re

@OPERATOR_REGISTRY.register()
class GeoKGEventTupleLocationFilter(OperatorABC):
    """
    Filter spatio-temporal event tuples based on a specific location.

    Event tuple format:
        "<event> ... <location> LocationName <time> ... <optional fields>"
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGEventTupleLocationFilter 根据指定地点筛选事件多元组。",
                "输入: List[str]\n输出: List[str>"
            )
        return (
            "KGEventTupleLocationFilter filters spatio-temporal event tuples by location.",
            "Input: List[str]\nOutput: List[str>"
        )

    # ------------------- 提取 location -------------------
    def _extract_location(self, tuple_str: str) -> Optional[str]:
        """
        提取 <location> 标签对应的值
        """
        m = re.search(r"<location>\s*(.*?)(\s*<|$)", tuple_str)
        if not m:
            return None
        return m.group(1).strip()

    # ------------------- 核心过滤 -------------------
    def _filter(self, tuples: List[str], location_name: str) -> List[str]:
        """
        只保留 location 字段中包含 location_name 的元组（模糊匹配，忽略大小写）
        """
        results = []
        for t in tuples:
            loc = self._extract_location(t)
            if loc and location_name.lower() in loc.lower():
                results.append(t)
        return results

    # ------------------- DataFrame 验证 -------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str):
        if input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {input_key}")

    # ------------------- 主运行方法 -------------------
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "tuple",
        output_key: str = "filtered_tuple",
        location_name: str = "China"
    ):
        self.input_key = input_key
        self.output_key = output_key
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key)
        self.logger.info(f"Filtering event tuples by location containing: {location_name}")

        results = []
        for row in df[input_key]:
            if not isinstance(row, list):
                results.append([])
                continue
            filtered = self._filter(row, location_name)
            results.append(filtered)

        if self.merge_to_input:
            df[self.input_key] = results
            output_file = storage.write(df)
            self.logger.info(f"Results saved to {output_file}")
            return [self.input_key]

        df[self.output_key] = results
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]