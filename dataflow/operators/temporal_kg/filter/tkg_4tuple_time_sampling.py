import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import re
from typing import List, Optional
from datetime import datetime, timedelta


@OPERATOR_REGISTRY.register()
class TKGTupleTimeFilter(OperatorABC):
    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "TKGTupleTimeFilter 用于根据给定的时间范围筛选时序知识图谱中的 tuple 或四元组，保留时间信息与查询时间区间相匹配的结果。",
                "输入: 数据表中需要包含一个 tuple 字段，通常由 input_key 指定，默认是 tuple。"
                "每一行输入通常是一个列表，列表中的元素为带有 <time> 标签的时序知识图谱字符串，支持关系四元组格式 "
                "\"<subj> A <obj> B <rel> R <time> T\"，也支持属性四元组格式 "
                "\"<entity> A <attribute> B <value> C <time> T\"。"
                "算子支持多种时间表达形式，包括具体日期（如 2021-05-01）、月份（如 March 2021）、年份（如 2023）、季度（如 Q1 2021）、"
                "季节（如 Summer 2022）以及时间区间（如 2021-01-01|2021-12-31）。"
                "用户可通过 query_time_start 和 query_time_end 指定查询时间范围，算子会解析每条 tuple 中的时间并判断其是否与查询区间重叠。"
                "输出: filtered_tuple 或覆盖原输入列。"
                "当 merge_to_input=False 时，筛选后的结果写入 output_key 指定的新字段，通常为 filtered_tuple；"
                "当 merge_to_input=True 时，筛选结果直接覆盖原始 input_key 字段。"
                "若某一行输入不是列表、tuple 中没有可解析的 <time> 字段，或其时间与查询区间不匹配，则该行对应位置不会保留该 tuple。",
            )
        return (
            "TKGTupleTimeFilter is used to filter temporal KG tuples or quadruples by a given time range, keeping only the results whose temporal information matches the query interval.",
            "Input: the dataframe must contain a tuple field specified by input_key, which defaults to tuple. "
            "Each row is usually a list whose elements are temporal KG strings containing a <time> tag. "
            "Supported formats include relation quadruples such as "
            "\"<subj> A <obj> B <rel> R <time> T\" and attribute quadruples such as "
            "\"<entity> A <attribute> B <value> C <time> T\". "
            "The operator supports multiple temporal expressions, including exact dates (e.g. 2021-05-01), months (e.g. March 2021), years (e.g. 2023), quarters (e.g. Q1 2021), "
            "seasons (e.g. Summer 2022), and time intervals (e.g. 2021-01-01|2021-12-31). "
            "Users specify the query interval through query_time_start and query_time_end, and the operator parses the time of each tuple and checks whether it overlaps with the query range. "
            "Output: filtered_tuple or overwrite the original input column. "
            "When merge_to_input=False, the filtered result is written to a new column specified by output_key, usually filtered_tuple; "
            "when merge_to_input=True, the filtered result overwrites the original input_key column. "
            "If a row is not a list, a tuple does not contain a parsable <time> field, or its time does not match the query interval, that tuple will not be kept in the output.",
        )

    def _extract_time(self, tuple_str: str) -> Optional[str]:
        m = re.search(r"<time>\s*(.*)", tuple_str)
        if not m:
            return None
        return m.group(1).strip()

    def _parse_date(self, t: str):
        if not t or t == "NA":
            return None
        formats = ["%Y-%m-%d", "%B %Y", "%b %Y", "%Y"]
        for f in formats:
            try:
                return datetime.strptime(t, f)
            except Exception:
                pass
        return None

    def _parse_quarter(self, t):
        m = re.match(r"Q([1-4])\s+(\d{4})", t)
        if not m:
            return None
        q = int(m.group(1))
        year = int(m.group(2))
        start_month = (q - 1) * 3 + 1
        start = datetime(year, start_month, 1)
        end = datetime(year, start_month + 2, 28)
        return start, end

    def _parse_month(self, t):
        try:
            dt = datetime.strptime(t, "%B %Y")
        except Exception:
            try:
                dt = datetime.strptime(t, "%b %Y")
            except Exception:
                return None
        year, month = dt.year, dt.month
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year, 12, 31)
        else:
            end = datetime(year, month + 1, 1) - timedelta(days=1)
        return start, end

    def _parse_season(self, t):
        m = re.match(r"(Spring|Summer|Autumn|Fall|Winter)\s+(\d{4})", t, re.I)
        if not m:
            return None
        season = m.group(1).lower()
        year = int(m.group(2))
        if season == "spring":
            return datetime(year, 3, 1), datetime(year, 5, 31)
        if season == "summer":
            return datetime(year, 6, 1), datetime(year, 8, 31)
        if season in ["autumn", "fall"]:
            return datetime(year, 9, 1), datetime(year, 11, 30)
        if season == "winter":
            return datetime(year, 12, 1), datetime(year + 1, 2, 28)
        return None

    def _parse_tuple_time(self, t):
        if t == "NA":
            return None, None
        if "|" in t:
            parts = t.split("|")
            try:
                return datetime.strptime(parts[0], "%Y-%m-%d"), datetime.strptime(parts[1], "%Y-%m-%d")
            except Exception:
                return None, None
        q = self._parse_quarter(t)
        if q:
            return q
        s = self._parse_season(t)
        if s:
            return s
        m = self._parse_month(t)
        if m:
            return m
        d = self._parse_date(t)
        if d:
            return d, d
        return None, None

    def _parse_query_time(self, t):
        if not t:
            return None, None
        q = self._parse_quarter(t)
        if q:
            return q
        s = self._parse_season(t)
        if s:
            return s
        m = self._parse_month(t)
        if m:
            return m
        d = self._parse_date(t)
        if d:
            return d, d
        return None, None

    def _match_time(self, tuple_start, tuple_end, query_start, query_end):
        if tuple_start is None:
            return False
        if query_start and not query_end:
            return tuple_end >= query_start
        if query_start and query_end:
            return not (tuple_end < query_start or tuple_start > query_end)
        return False

    def _filter(self, tuples: List[str], query_start, query_end) -> List[str]:
        results = []
        for t in tuples:
            tuple_time = self._extract_time(t)
            if not tuple_time:
                continue
            start, end = self._parse_tuple_time(tuple_time)
            if self._match_time(start, end, query_start, query_end):
                results.append(t)
        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column '{self.output_key}' already exists")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "tuple",
        output_key: str = "filtered_tuple",
        query_time_start: str = "Q1 2021",
        query_time_end: str = "2023",
    ):
        self.input_key = input_key
        self.output_key = output_key
        df = storage.read("dataframe")
        self._validate_dataframe(df)
        self.logger.info("Filtering tuples by temporal constraints")

        query_start, _ = self._parse_query_time(query_time_start)
        query_end = None
        if query_time_end:
            _, query_end = self._parse_query_time(query_time_end)

        results = []
        for row in df[input_key]:
            if not isinstance(row, list):
                results.append([])
                continue
            filtered = self._filter(row, query_start, query_end)
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