from dataflow.prompts.diverse_kg.geokg import GeoKGRelationInferencePrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow.core import OperatorABC, LLMServingABC
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import json
import re


@OPERATOR_REGISTRY.register()
class GeoKGRelationInference(OperatorABC):
    """
    Given two entities and a KG, infer the relation between them using related
    quadruples and a predefined ontology. Calls LLM for reasoning.
    """

    def __init__(self, llm_serving: LLMServingABC, lang: str = "en", seed: int = 0):
        self.llm_serving = llm_serving
        self.lang = lang
        self.rng = seed
        self.logger = get_logger()
        self.prompt_template = GeoKGRelationInferencePrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "GeoKGRelationInference 给定两个实体和 KG 四元组，推断它们之间的关系",
                "输入: entity_pair, tuple, ontology; 输出: 推断的四元组"
            )
        else:
            return (
                "GeoKGRelationInference infers relation between two entities from KG tuples.",
                "Input: entity_pair, tuple, ontology; Output: inferred quadruple"
            )

    def run(
        self,
        storage: DataFlowStorage = None,
        entity_pair: List[str] = ["Hubei", "China"],
        input_key_tuple: str = "tuple",
        input_key_meta: str = "ontology",
        output_key: str = "inferred_tuple"
    ) -> List[str]:

        dataframe = storage.read("dataframe")

        if entity_pair is None or len(entity_pair) != 2:
            raise ValueError("entity_pair must be a list of 2 entities, e.g. ['Wuhan','China']")

        tuples_list = dataframe[input_key_tuple].tolist()

        # 读取 ontology
        storage_meta = FileStorage(first_entry_file_name="", cache_type="json")
        ontology_df = storage_meta.read(file_path=f"./.cache/api/{input_key_meta}.json", output_type="dataframe")
        row = ontology_df.iloc[0]
        ontology_lists = {
            "entity_type": row["entity_type"],
            "relation_type": row["relation_type"],
            "attribute_type": row.get("attribute_type", {})
        }

        outputs = self.process_batch(tuples_list, entity_pair, ontology_lists)
        dataframe[output_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    # -------------------------------
    # 核心处理
    # -------------------------------
    def process_batch(
        self,
        tuples_list: List[List[str]],
        entity_pair: List[str],
        ontology_lists: dict
    ) -> List[List[str]]:

        raw_data = [{"tuples": tuples} for tuples in tuples_list]

        return self._construct_examples(raw_data, entity_pair, ontology_lists)

    def _construct_examples(
        self,
        raw_data: List[Dict[str, Any]],
        entity_pair: List[str],
        ontology_lists: dict
    ) -> List[List[str]]:

        self.logger.info("Starting relation inference...")
        results = []

        e1, e2 = entity_pair

        for data in tqdm(raw_data, desc="Infer relation"):
            tuples = data.get("tuples", [])

            related_tuples = self._find_related_tuples(tuples, e1, e2)
            if not related_tuples:
                results.append([])
                continue

            # 构建 LLM 输入 prompt
            user_prompt = self.prompt_template.build_prompt(
                entity1=e1,
                entity2=e2,
                tuples=related_tuples
            )

            system_prompt = self.prompt_template.build_system_prompt(ontology_lists)

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )

            inferred_tuple = self._parse_llm_response(responses[0])
            results.append(inferred_tuple)

        return results

    # -------------------------------
    # 相关辅助函数
    # -------------------------------
    def _find_related_tuples(
        self,
        tuples: List[str],
        entity1: str,
        entity2: str
    ) -> List[str]:

        related = []
        for t in tuples:
            subj_match = re.search(r"<subj> (.*?) <obj>", t)
            obj_match = re.search(r"<obj> (.*?) <rel>", t)

            subj = subj_match.group(1) if subj_match else ""
            obj = obj_match.group(1) if obj_match else ""

            if subj in [entity1, entity2] or obj in [entity1, entity2]:
                related.append(t)

        return related

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        解析 LLM 输出 JSON 中的 tuple
        """
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("tuple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []