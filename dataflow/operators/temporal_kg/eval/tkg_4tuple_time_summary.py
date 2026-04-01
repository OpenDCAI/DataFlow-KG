import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import re
from typing import List, Optional
from datetime import datetime


@OPERATOR_REGISTRY.register()
class TKGTemporalStatistics(OperatorABC):
    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "TKGTemporalStatistics 用于统计时序知识图谱 tuple 中的时间信息分布，包括时间字段非空比例以及不同年份的时间分布情况。",
                "输入: 数据表中需要包含一个 tuple 字段，通常由 input_key 指定，默认是 tuple。"
                "每一行输入通常是一个列表，列表中的元素为带有 <time> 标签的时序知识图谱字符串，支持关系四元组格式 "
                "\"<subj> A <obj> B <rel> R <time> T\"，也支持属性四元组格式 "
                "\"<entity> A <attribute> B <value> C <time> T\"。"
                "算子会逐条抽取其中的 <time> 字段，统计总 tuple 数、有效时间 tuple 数、非 NA 时间比例，并进一步解析年份信息，"
                "生成按年份归一化后的时间分布。"
                "输出: temporal_statistics。该字段为字典，通常包含 total_tuples、valid_time_tuples、non_na_ratio 和 year_distribution 四项；"
                "其中 year_distribution 是年份到比例的映射字典。若某一行输入不是列表、列表为空，或没有可统计的时间信息，则输出空字典。",
            )
        return (
            "TKGTemporalStatistics is used to compute temporal statistics from temporal KG tuples, including the ratio of non-empty time fields and the distribution over years.",
            "Input: the dataframe must contain a tuple field specified by input_key, which defaults to tuple. "
            "Each row is usually a list whose elements are temporal KG strings containing a <time> tag. "
            "Supported formats include relation quadruples such as "
            "\"<subj> A <obj> B <rel> R <time> T\" and attribute quadruples such as "
            "\"<entity> A <attribute> B <value> C <time> T\". "
            "The operator extracts the <time> field from each tuple, computes the total number of tuples, the number of tuples with valid time values, "
            "the non-NA time ratio, and further parses the year information to produce a normalized yearly distribution. "
            "Output: temporal_statistics. This field is a dictionary that usually contains total_tuples, valid_time_tuples, non_na_ratio, and year_distribution; "
            "year_distribution is a dictionary mapping years to normalized ratios. "
            "If a row is not a list, the list is empty, or no usable temporal information can be collected, an empty dictionary is returned for that row.",
        )

    def _extract_time(self, tuple_str: str) -> Optional[str]:
        m = re.search(r"<time>\s*(.*)", tuple_str)
        if not m:
            return None
        return m.group(1).strip()

    def _parse_year(self, t: str) -> Optional[int]:
        if not t or t == "NA":
            return None

        if "|" in t:
            try:
                start = datetime.strptime(t.split("|")[0], "%Y-%m-%d")
                return start.year
            except Exception:
                return None

        formats = ["%Y-%m-%d", "%B %Y", "%b %Y", "%Y"]
        for f in formats:
            try:
                dt = datetime.strptime(t, f)
                return dt.year
            except Exception:
                pass

        m = re.match(r"Q([1-4])\s+(\d{4})", t)
        if m:
            return int(m.group(2))

        m = re.match(r"(Spring|Summer|Autumn|Fall|Winter)\s+(\d{4})", t, re.I)
        if m:
            return int(m.group(2))

        return None

    def _collect_year_statistics(self, tuples: List[str]):
        total = 0
        valid_time = 0
        year_count = {}

        for t in tuples:
            total += 1
            time_str = self._extract_time(t)

            if not time_str or time_str == "NA":
                continue

            valid_time += 1
            year = self._parse_year(time_str)

            if year is None:
                continue

            year_count[year] = year_count.get(year, 0) + 1

        return total, valid_time, year_count

    def _compute_year_ratio(self, year_count: dict, valid_time: int):
        if valid_time == 0:
            return {}

        result = {}
        for year, count in sorted(year_count.items()):
            result[year] = count / valid_time

        return result

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")

        if self.output_key in dataframe.columns:
            raise ValueError(f"Column '{self.output_key}' already exists")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "tuple",
        output_key: str = "temporal_statistics",
    ):
        self.input_key = input_key
        self.output_key = output_key

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        self.logger.info("Computing temporal statistics")

        results = []

        for row in df[input_key]:
            if not isinstance(row, list):
                results.append({})
                continue

            total, valid_time, year_count = self._collect_year_statistics(row)

            if total == 0:
                results.append({})
                continue

            non_na_ratio = valid_time / total
            year_ratio = self._compute_year_ratio(year_count, valid_time)

            results.append({
                "total_tuples": total,
                "valid_time_tuples": valid_time,
                "non_na_ratio": non_na_ratio,
                "year_distribution": year_ratio
            })

        df[self.output_key] = results

        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]