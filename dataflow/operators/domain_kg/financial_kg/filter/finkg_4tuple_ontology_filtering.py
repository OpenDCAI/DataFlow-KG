# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: FinKGTupleFilter
====================================

License:
    MIT License
"""

import re
from typing import Any, Dict, List, Optional

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class FinKGTupleFilter(OperatorABC):

    def __init__(self, ontology_list: List[Dict[str, Any]] = None):
        self.ontology = ontology_list[0] if isinstance(ontology_list, list) else ontology_list
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGTupleFilter 用于根据目标金融本体筛选四元组。",
                "输入: tuple + entity_class + ontology; 输出: filtered_tuple",
            )
        return (
            "FinKGTupleFilter is used to filter Financial KG tuples by target ontology.",
            "Input: tuple + entity_class + ontology; Output: filtered_tuple",
        )

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists: Optional[List[Dict[str, Any]]] = None,
        input_key_tuple: str = "tuple",
        input_key_class: str = "entity_class",
        output_key: str = "filtered_tuple",
        input_key_meta: str = "finkg_ontology",
        target_ontology: str = "Corporation"
    ):
        dataframe = storage.read("dataframe")
        self.ontology = load_finkg_ontology(
            ontology_lists=ontology_lists,
            input_key_meta=input_key_meta,
        )

        tuples_list = dataframe[input_key_tuple].tolist()
        class_list = dataframe[input_key_class].tolist()

        filtered_results = []

        for tuples, classes in zip(tuples_list, class_list):
            filtered = self._filter_tuples(
                tuples=tuples,
                entity_classes=classes,
                target_ontology=target_ontology
            )
            filtered_results.append(filtered)

        dataframe[output_key] = filtered_results
        output_file = storage.write(dataframe)
        self.logger.info(f"Filtered tuples saved to {output_file}")

        return [output_key]

    # ------------------------------------------------

    def _get_target_type(self, target: str) -> Dict[str, Any]:

        if not self.ontology:
            raise ValueError("ontology must not be empty")

        for _, attrs in self.ontology.get("attribute_type", {}).items():
            if target in attrs:
                return {"type": "attribute_type"}

        for _, rels in self.ontology.get("relation_type", {}).items():
            if target in rels:
                return {"type": "relation_type"}

        for _, ents in self.ontology.get("entity_type", {}).items():
            if target in ents:
                return {"type": "entity_type"}

        raise ValueError(f"Target '{target}' not found in ontology")

    def _detect_tuple_type(self, tuple_str: str) -> str:

        if "<rel>" in tuple_str:
            return "ER"

        if "<attribute>" in tuple_str:
            return "EA"

        return "UNKNOWN"

    # ------------------------------------------------

    def _filter_tuples(
        self,
        tuples: List[str],
        entity_classes: List[List[str]],
        target_ontology: str
    ) -> List[str]:

        target_info = self._get_target_type(target_ontology)

        filtered = []

        for t, cls in zip(tuples, entity_classes):

            t_type = self._detect_tuple_type(t)

            # attribute_type 过滤 (EA)
            if target_info["type"] == "attribute_type" and t_type == "EA":
                attr_match = re.search(r"<attribute> (.*?) <value>", t)
                attr = attr_match.group(1) if attr_match else ""
                if attr == target_ontology:
                    filtered.append(t)

            # relation_type 过滤 (ER)
            elif target_info["type"] == "relation_type" and t_type == "ER":
                rel_match = re.search(r"<rel> (.*?) <time>", t)
                rel = rel_match.group(1) if rel_match else ""
                if rel == target_ontology:
                    filtered.append(t)

            # entity_type 过滤 (EA 或 ER 均可)
            elif target_info["type"] == "entity_type":
                if isinstance(cls, list) and target_ontology in cls:
                    filtered.append(t)

        return filtered
