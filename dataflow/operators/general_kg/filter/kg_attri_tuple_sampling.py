import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

import random
from typing import List, Dict
import re
from collections import defaultdict


@OPERATOR_REGISTRY.register()
class KGAttributeTupleSampler(OperatorABC):
    """
    对实体-属性-属性值(-其他字段)三元组/多元组进行采样分组。

    支持两种模式：
        1. group_by="entity"
        2. group_by="attribute"

    输入示例：
        "<entity> Henry <attr> occupation <value> singer <time> 2018"

    输出：
        [
            {"subgraph": [...]},
            {"subgraph": [...]}
        ]
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        seed: int = 0,
        lang: str = "en",
        group_by: str = "entity",  # entity / attribute
        max_groups: int = None,
        max_per_group: int = None,
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.group_by = group_by
        self.max_groups = max_groups
        self.max_per_group = max_per_group
        self.logger = get_logger()

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGAttributeTupleSampler 用于对实体属性多元组进行分组采样。",
                "支持按实体或按属性分组，可限制组数和每组元组数量。",
                "输入列 tuple（或 triple）为属性多元组字符串列表，输出列 subgraph 为按分组采样后的元组列表，每行对应一个分组。"
            )
        else:
            return (
                "KGAttributeTupleSampler groups and samples attribute tuples by entity or attribute.",
                "Supports entity-based or attribute-based grouping with optional limits on group count and group size.",
                "Takes tuple (or triple) as input column containing attribute tuple strings, and outputs subgraph (List[str] per group of sampled tuples)."
            )

    # =========================
    # Parsing
    # =========================
    def _parse_tuple(self, t: str):
        """
        解析实体-属性-属性值-可扩展字段

        只提取：
            entity
            attribute
        其他字段保留原始字符串
        """

        entity_match = re.search(r"<entity>\s*(.+?)\s*(?=<attribute>)", t)
        attr_match = re.search(r"<attribute>\s*(.+?)\s*(?=<value>)", t)

        if not entity_match or not attr_match:
            return None

        entity = entity_match.group(1).strip()
        attribute = attr_match.group(1).strip()

        return {
            "entity": entity,
            "attribute": attribute,
            "raw": t.strip(),
        }

    # =========================
    # Core Logic
    # =========================
    def _group_and_sample(self, tuples: List[str]) -> List[Dict]:
        grouped = defaultdict(list)

        for t in tuples:
            parsed = self._parse_tuple(t)
            if not parsed:
                continue

            if self.group_by == "entity":
                key = parsed["entity"]
            elif self.group_by == "attribute":
                key = parsed["attribute"]
            else:
                raise ValueError("group_by must be 'entity' or 'attribute'")

            grouped[key].append(parsed["raw"])

        # 转为列表
        groups = list(grouped.values())

        # 可选：限制组数量
        if self.max_groups:
            self.rng.shuffle(groups)
            groups = groups[: self.max_groups]

        results = []

        for g in groups:
            if self.max_per_group:
                self.rng.shuffle(g)
                g = g[: self.max_per_group]

            results.append({
                "subgraph": g
            })

        return results

    # =========================
    # DataFrame Interface
    # =========================
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if hasattr(self, "input_key") and self.input_key in dataframe.columns:
            return
        elif "tuple" in dataframe.columns:
            self.input_key = "tuple"
        elif "triple" in dataframe.columns:
            self.input_key = "triple"
        else:
            raise ValueError(
                "Missing required input column: neither 'tuple' nor 'triple' found"
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "tuple",
        output_key: str = "subgraph",
    ):
        self.input_key = input_key
        df = storage.read("dataframe")

        self._validate_dataframe(df)

        self.logger.info(
            f"Sampling tuples grouped by {self.group_by}"
        )

        # =========================
        # NEW: 合并所有数据
        # =========================

        all_tuples = []

        if len(df) == 0:
            raise ValueError("DataFrame is empty.")

        elif len(df) == 1:
            # 只有一行
            row_data = df[self.input_key].iloc[0]
            if isinstance(row_data, list):
                all_tuples = row_data
            else:
                raise ValueError("Row data must be List[str]")

        else:
            # 多行 → 合并
            for row in df[self.input_key]:
                if isinstance(row, list):
                    all_tuples.extend(row)

        # =========================
        # 统一处理
        # =========================
        
        grouped = self._group_and_sample(all_tuples)

        # 整个 df 只写一份结果
        data = pd.DataFrame()
        data[output_key] = [o[output_key] for o in grouped]

        output_file = storage.write(data)
        self.logger.info(f"Subgraphs saved to {output_file}")

        return [output_key]