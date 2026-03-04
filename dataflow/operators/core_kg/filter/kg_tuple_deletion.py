import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import random
from typing import List
import re

@OPERATOR_REGISTRY.register()
class KGTupleDeletion(OperatorABC):
    """
    KGTupleDeletion removes knowledge graph tuples related to specified entities.

    Supports:
    - Entity–relation–entity triples
    - Entity–attribute–value triples
    - Higher-order tuples (n-tuples) with arbitrary additional fields
    """

    def __init__(
        self,
        llm_serving=None,
        seed: int = 0,
        lang: str = "en",
        merge_to_input: bool = True
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

        # Pattern to match all <tag> value fields in a tuple
        self.tag_value_pattern = re.compile(r"<(.*?)>\s*(.+?)\s*(?=<|$)")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTupleDeletion 用于删除与指定实体相关的知识图谱元组。",
                "该算子支持三元组、四元组及更多元组的删除。",
                "当元组中的任意实体命中目标实体列表时，该元组将被移除。",
            )
        else:
            return (
                "KGTupleDeletion removes tuples related to specified entities.",
                "Supports triples, quadruples, and higher-order tuples.",
                "Tuples are removed if any entity field matches the target entity list.",
            )

    def _remove_entity_related_tuples(
        self,
        entity_list: List[str],
        tuple_2d_list: List[List[str]],
    ) -> List[List[str]]:
        """
        Remove tuples (triples, quadruples, or n-tuples) involving target entities.
        """
        target_entities = {e.strip().lower() for e in entity_list}
        cleaned_result = []

        for tuple_list in tuple_2d_list:
            current_clean = []

            for tup in tuple_list:
                # Extract all <tag> values
                matches = self.tag_value_pattern.findall(tup)
                # Find all values corresponding to entity-like tags
                entity_values = [v for tag, v in matches if tag.lower() in ["subj", "obj", "entity"]]

                # If any entity in the tuple matches target, skip it
                if any(ev.strip().lower() in target_entities for ev in entity_values):
                    continue

                current_clean.append(tup)

            cleaned_result.append(current_clean)

        return cleaned_result

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        # Determine input column dynamically: 'triple' or 'tuple'
        if "triple" in dataframe.columns:
            self.input_key = "triple"
            self.output_key = "new_triple" if not self.merge_to_input else "triple"
        elif "tuple" in dataframe.columns:
            self.input_key = "tuple"
            self.output_key = "normalized_tuple" if not self.merge_to_input else "tuple"
        else:
            raise ValueError("MISSING: No 'triple' or 'tuple' column found in dataframe")

        if self.output_key in dataframe.columns and not self.merge_to_input:
            raise ValueError(
                f"Column '{self.output_key}' already exists and would be overwritten"
            )

    def run(
        self,
        storage: DataFlowStorage,
        target_entity: List[str] = ["Tesla Model Y"]
    ):
        """
        Execute entity-based tuple deletion.
        """
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        tuples = dataframe[self.input_key].tolist()

        cleaned_tuples = self._remove_entity_related_tuples(
            target_entity,
            tuples,
        )

        if self.merge_to_input:
            dataframe[self.input_key] = cleaned_tuples
            output_file = storage.write(dataframe)
            self.logger.info(f"Results saved to {output_file}")
            return [self.input_key]
        else:
            dataframe[self.output_key] = cleaned_tuples
            output_file = storage.write(dataframe)
            self.logger.info(f"Results saved to {output_file}")
            return [self.output_key]