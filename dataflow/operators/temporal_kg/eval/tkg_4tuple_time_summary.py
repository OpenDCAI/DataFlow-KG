# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: TKGTemporalStatistics
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-03-05
"""

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

    """
    Statistics for temporal knowledge graph tuples.

    Supported tuple formats:

    Relation quadruple:
        "<subj> A <obj> B <rel> R <time> T"

    Attribute quadruple:
        "<entity> A <attribute> B <value> C <time> T"
    """

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTemporalStatistics 统计知识图谱中的时间信息。",
                "包括时间非NA比例和不同年份时间分布比例。",
                "Input: List[str]\nOutput: Dict",
            )
        return (
            "KGTemporalStatistics computes temporal statistics of KG tuples.",
            "Including non-NA ratio and yearly distribution.",
            "Input: List[str]\nOutput: Dict",
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
            except:
                return None

        formats = ["%Y-%m-%d","%B %Y","%b %Y","%Y"]
        for f in formats:
            try:
                dt = datetime.strptime(t, f)
                return dt.year
            except:
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