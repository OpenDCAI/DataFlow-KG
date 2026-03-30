import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow import get_logger
import re
from typing import Any, Dict, List, Optional


@OPERATOR_REGISTRY.register()
class MedKGTripleFilter(OperatorABC):

    def __init__(self, ontology_list: List[Dict[str, Any]] = None):
        self.ontology_list = ontology_list
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") :
        if lang == "zh":
            return (
                "MedKGTripleFilter 用于根据目标本体筛选三元组",
                "输入: triple 列表; 输出: filtered_triple"
            )
        else:
            return (
                "MedKGTripleFilter filters KG triple based on target ontology",
                "Input: triple list; Output: filtered_triple"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key_triple, self.input_key_class]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists: Optional[List[Dict[str, Any]]] = None,
        input_key_triple: str = "triple",
        input_key_class: str = "entity_class",
        output_key: str = "filtered_triple",
        input_key_meta: str = "ontology",
        target_ontology: Optional[str] = None
    ):
        self.input_key_triple = input_key_triple
        self.input_key_class = input_key_class
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        if not target_ontology:
            raise ValueError("target_ontology must not be empty")

        # 加载 ontology
        if ontology_lists is None:

            storage_meta = FileStorage(first_entry_file_name="", cache_type="json")
            ontology_file_name = input_key_meta if input_key_meta.endswith(".json") else f"{input_key_meta}.json"

            ontology_df = storage_meta.read(
                file_path=f"./.cache/medical/{ontology_file_name}",
                output_type="dataframe"
            )

            row = ontology_df.iloc[0]

            ontology_lists = [{
                "entity_type": row["entity_type"],
                "relation_type": row["relation_type"],
                "attribute_type": row.get("attribute_type", {})
            }]

        self.ontology_list = ontology_lists

        triples_list = dataframe[input_key_triple].tolist()
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

        for idx, t in enumerate(triples):
            cls = entity_classes[idx] if idx < len(entity_classes) else []

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

                rel_match = re.search(r"<rel>\s*(.+?)$", t)
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
