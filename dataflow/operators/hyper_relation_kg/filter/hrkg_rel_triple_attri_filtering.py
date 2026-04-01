import re
from typing import List, Dict

import pandas as pd

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class HRKGRelationTripleAttributeFilter(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, lang: str = "en"):
        self.lang = lang
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "HRKGRelationTripleAttributeFilter 用于按照指定属性标签筛选知识图谱三元组或事件字符串，保留包含目标属性标签的 tuple。",
                "输入: 包含 tuple 字段的数据，输入字段通常是一个列表，其中每个元素是形如 "
                "\"<subj> ... <obj> ... <rel> ... <Location> ... <Time> ...\" 的三元组或事件字符串；"
                "同时需要提供 attr_tag 参数，用于指定目标属性标签，例如 <Location>、<Time>、<Value> 等。"
                "算子会逐行检查 tuple 列表，筛选出所有包含该属性标签的字符串。"
                "输出: filtered_tuple。该字段为筛选后的列表，只保留原输入中包含指定属性标签的 tuple；"
                "若某行输入不是列表或没有任何 tuple 命中该属性标签，则输出空列表。",
            )
        return (
            "HRKGRelationTripleAttributeFilter is used to filter KG triples or tuple-like event strings by a specified attribute tag, keeping only tuples that contain the target attribute.",
            "Input: a dataset containing a tuple field, where the input field is usually a list and each element is a triple or event-like string in a format such as "
            "\"<subj> ... <obj> ... <rel> ... <Location> ... <Time> ...\". "
            "An additional attr_tag parameter is required to specify the target attribute tag, such as <Location>, <Time>, <Value>, etc. "
            "The operator scans each row of the tuple list and retains only the strings that contain the specified attribute tag. "
            "Output: filtered_tuple. This field is a filtered list that keeps only the tuples containing the target attribute tag from the original input; "
            "if a row is not a list or no tuple matches the given attribute tag, an empty list is returned.",
        )

    def _parse_triple(self, triple_str: str) -> Dict:
        triple_str = triple_str.strip()

        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        if not subj_match:
            raise ValueError(f"No <subj> found in triple: {triple_str}")
        subj = subj_match.group(1).strip()

        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        if not obj_match:
            raise ValueError(f"No <obj> found in triple: {triple_str}")
        obj = obj_match.group(1).strip()

        rel_match = re.search(r"<rel>\s*(.+)$", triple_str)
        if not rel_match:
            raise ValueError(f"No <rel> found in triple: {triple_str}")
        full_rel = rel_match.group(1).strip()

        return {
            "subj": subj,
            "obj": obj,
            "rel": full_rel,
            "raw": triple_str,
        }

    def _filter_triples_by_attr(self, triples: List[str], attr_tag: str) -> List[str]:
        return [t for t in triples if attr_tag in t]

    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str):
        if input_key not in dataframe.columns:
            raise ValueError(f"Input column '{input_key}' not found in dataframe")
        self.input_key = input_key

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "tuple",
        output_key: str = "filtered_tuple",
        attr_tag: str = "<Location>"
    ):
        self._validate_dataframe(storage.read("dataframe"), input_key)
        df = storage.read("dataframe")

        filtered_triples_all = []

        for row in df[input_key]:
            if isinstance(row, list):
                filtered = self._filter_triples_by_attr(row, attr_tag)
            else:
                filtered = []
            filtered_triples_all.append(filtered)

        df[output_key] = filtered_triples_all

        output_file = storage.write(df)
        self.logger.info(f"Filtered triples saved to {output_file}")

        return [output_key]