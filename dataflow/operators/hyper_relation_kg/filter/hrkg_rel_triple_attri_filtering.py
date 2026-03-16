# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KGTripleAttributeFilter
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-03-16

License:
    MIT License
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

import re
from typing import List, Dict


@OPERATOR_REGISTRY.register()
class HRKGRelationTripleAttributeFilter(OperatorABC):
    """
    Filter triples by the presence of a specific attribute tag.
    
    Example attribute tags: "<Location>", "<Time>", "<Value>"
    """

    def __init__(self, llm_serving: LLMServingABC = None, lang: str = "en"):
        self.lang = lang
        self.logger = get_logger()

    # =========================
    # Triple parser (保留原逻辑)
    # =========================
    def _parse_triple(self, triple_str: str) -> Dict:
        triple_str = triple_str.strip()

        # 找 <subj>
        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        if not subj_match:
            raise ValueError(f"No <subj> found in triple: {triple_str}")
        subj = subj_match.group(1).strip()

        # 找 <obj>
        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        if not obj_match:
            raise ValueError(f"No <obj> found in triple: {triple_str}")
        obj = obj_match.group(1).strip()

        # 找 <rel> 到字符串末尾
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

    # =========================
    # 根据属性筛选 triples
    # =========================
    def _filter_triples_by_attr(self, triples: List[str], attr_tag: str) -> List[str]:
        """
        Filter triples that contain a specific attribute tag.
        """
        return [t for t in triples if attr_tag in t]

    # =========================
    # DataFrame 验证
    # =========================
    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str):
        if input_key not in dataframe.columns:
            raise ValueError(f"Input column '{input_key}' not found in dataframe")
        self.input_key = input_key

    # =========================
    # Run
    # =========================
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