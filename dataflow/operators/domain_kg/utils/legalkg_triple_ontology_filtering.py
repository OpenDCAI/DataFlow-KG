import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow import get_logger
import re
from typing import Any, Dict, List, Optional


@OPERATOR_REGISTRY.register()
class LegalKGTripleFilter(OperatorABC):

    def __init__(self, ontology_list: List[Dict[str, Any]] = None):
        self.ontology_list = ontology_list
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "GeoKGtripleFilter 用于根据目标本体筛选四元组",
                "输入: triple 列表; 输出: filtered_triple"
            )
        else:
            return (
                "GeoKGtripleFilter filters KG triples based on target ontology",
                "Input: triple list; Output: filtered_triple"
            )

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists: Optional[List[Dict[str, Any]]] = None,
        input_key: str = "triple",
        input_key_class: str = "entity_class",
        output_key: str = "filtered_triple",
        input_key_meta: str = "legal_ontology",
        target_ontology: str = "自然人"
    ):

        dataframe = storage.read("dataframe")

        # 加载 ontology
        if ontology_lists is None:

            storage_meta = FileStorage(first_entry_file_name="", cache_type="json")

            ontology_df = storage_meta.read(
                file_path=f"./.cache/api/{input_key_meta}.json",
                output_type="dataframe"
            )

            row = ontology_df.iloc[0]

            ontology_lists = [{
                "entity_type": row["entity_type"],
                "relation_type": row["relation_type"],
                "attribute_type": row.get("attribute_type", {})
            }]

        self.ontology_list = ontology_lists

        triples_list = dataframe[input_key].tolist()
        class_list = dataframe[input_key_class].tolist()

        filtered_results = []

        for triples, classes in zip(triples_list, class_list):

            filtered = self._filter_triples(
                triples=triples,
                entity_classes=classes,
                target_ontology=target_ontology
            )

            filtered_results.append(filtered)

        dataframe[output_key] = filtered_results

        output_file = storage.write(dataframe)

        self.logger.info(f"Filtered triples saved to {output_file}")

        return [output_key]

    # ------------------------------------------------

    def _get_target_type(self, target: str) -> Dict[str, Any]:

        if not self.ontology_list:
            raise ValueError("ontology_list must not be empty")

        ontology = self.ontology_list[0]

        for _, attrs in ontology.get("attribute_type", {}).items():
            if target in attrs:
                return {"type": "attribute_type"}

        for _, rels in ontology.get("relation_type", {}).items():
            if target in rels:
                return {"type": "relation_type"}

        for _, ents in ontology.get("entity_type", {}).items():
            if target in ents:
                return {"type": "entity_type"}

        raise ValueError(f"Target '{target}' not found in ontology")

    def _detect_triple_type(self, triple_str: str) -> str:

        if "<rel>" in triple_str:
            return "ER"

        if "<attribute>" in triple_str:
            return "EA"

        return "UNKNOWN"

    # ------------------------------------------------

    def _filter_triples(
        self,
        triples: List[str],
        entity_classes: List[List[str]],
        target_ontology: str
    ) -> List[str]:

        target_info = self._get_target_type(target_ontology)

        filtered = []

        for t, cls in zip(triples, entity_classes):

            t_type = self._detect_triple_type(t)

            # -----------------------------
            # attribute_type 过滤 (EA)
            # -----------------------------
            if target_info["type"] == "attribute_type" and t_type == "EA":

                attr_match = re.search(r"<attribute> (.*?) <value>", t)
                attr = attr_match.group(1) if attr_match else ""

                if attr == target_ontology:
                    filtered.append(t)

            # -----------------------------
            # relation_type 过滤 (ER)
            # -----------------------------
            elif target_info["type"] == "relation_type" and t_type == "ER":

                rel_match = re.search(r"<rel> (.*?) <time>", t)
                rel = rel_match.group(1) if rel_match else ""

                if rel == target_ontology:
                    filtered.append(t)

            # -----------------------------
            # entity_type 过滤
            # EA: 主体实体类型
            # ER: 两个实体任意一个
            # -----------------------------
            elif target_info["type"] == "entity_type":

                if isinstance(cls, list) and target_ontology in cls:
                    filtered.append(t)

        return filtered