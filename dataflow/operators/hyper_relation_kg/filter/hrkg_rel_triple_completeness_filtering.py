import pandas as pd
from typing import List, Optional

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class HRKGTripleCompletenessFilter(OperatorABC):
    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "HRKGTripleCompletenessFilter 用于根据三元组对应的评分列表筛选知识图谱三元组，保留分数落在指定区间内的 tuple。",
                "输入: 数据表中需包含一个三元组列表字段和一个与之位置对齐的分数列表字段。"
                "其中 input_key 对应三元组列表，通常为 tuple 字段，每个元素是一条三元组或事件字符串；"
                "score_key 对应评分列表，通常为 consistency_scores 或其他逐条评分结果，其长度应与三元组列表一致。"
                "算子会按位置一一对齐三元组和分数，仅保留分数满足 min_score <= score <= max_score 的三元组。"
                "输出: filtered_tuple 或覆盖原输入列。"
                "当 merge_to_input=False 时，输出写入 output_key 指定的新字段，通常为 filtered_tuple；"
                "当 merge_to_input=True 时，直接用筛选结果覆盖原 input_key 字段。"
                "若某行中的三元组列或评分列不是列表，则该行输出为空列表。",
            )
        return (
            "HRKGTripleCompletenessFilter is used to filter KG triples based on an aligned list of scores, keeping only tuples whose scores fall within a specified range.",
            "Input: the dataframe must contain one column storing a list of triples and another column storing an aligned list of scores. "
            "The input_key usually corresponds to a tuple field, where each element is a triple string or an event-like tuple expression; "
            "the score_key corresponds to a score list, typically consistency_scores or another per-tuple evaluation result, and its length is expected to be aligned with the triple list. "
            "The operator matches triples and scores by position and keeps only the triples satisfying min_score <= score <= max_score. "
            "Output: filtered_tuple or overwrite the original input column. "
            "When merge_to_input=False, the filtered result is written to a new column specified by output_key, usually filtered_tuple; "
            "when merge_to_input=True, the filtered result overwrites the original input_key column. "
            "If either the triple column or the score column in a row is not a list, an empty list is returned for that row.",
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
        score_key: str = "completeness_scores",
        output_key: str = "filtered_tuple",
        min_score: float = 0.95,
        max_score: float = 1.0,
    ):
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