# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: FinKGRelationChainInference
====================================

License:
    MIT License
"""

import json
import re
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from dataflow.core import LLMServingABC, OperatorABC
from dataflow.prompts.diverse_kg.finkg import FinKGRelationChainInferencePrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow import get_logger


# ======================================================================
# Helper class — all business logic lives here
# ======================================================================

class FinKGRelationChainInferenceEngine:
    """
    LLM-based multi-hop relation chain inference engine.

    Given a target entity pair and a set of relation quadruples,
    selects a k-hop neighborhood around the pair, sends context
    to LLM for reasoning, and parses the inferred results.
    """

    _REL_PATTERN = re.compile(
        r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*?)\s*<time>\s*(.*)"
    )

    def __init__(self, k_hops: int = 2, logger=None):
        self.k_hops = max(1, int(k_hops))
        self.logger = logger or get_logger()

    # ------------------------------------------------------------------
    # Tuple parsing
    # ------------------------------------------------------------------

    @classmethod
    def _parse_tuple(cls, tuple_str: str) -> Optional[Dict[str, str]]:
        if not isinstance(tuple_str, str):
            return None
        m = cls._REL_PATTERN.search(tuple_str)
        if not m:
            return None
        subj, obj_, rel = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if not subj or not obj_ or not rel:
            return None
        return {
            "subj": subj,
            "obj": obj_,
            "rel": rel,
            "time": (m.group(4).strip() or "NA"),
            "raw": tuple_str,
        }

    # ------------------------------------------------------------------
    # k-hop BFS neighborhood selection
    # ------------------------------------------------------------------

    def _find_related_tuples(
        self,
        tuples: List[str],
        entity1: str,
        entity2: str,
    ) -> List[str]:
        """
        BFS from {entity1, entity2} over the tuple graph for k hops.
        Returns raw tuple strings touched by the expanded frontier.
        """
        if not isinstance(tuples, list) or not tuples:
            return []

        edges: List[Dict[str, str]] = []
        for t in tuples:
            rec = self._parse_tuple(t)
            if rec:
                edges.append(rec)

        if not edges:
            return []

        # Build entity → edge-index mapping for fast lookup
        ent_to_eids: Dict[str, List[int]] = {}
        for i, e in enumerate(edges):
            ent_to_eids.setdefault(e["subj"], []).append(i)
            ent_to_eids.setdefault(e["obj"], []).append(i)

        frontier = {entity1, entity2}
        visited = set(frontier)
        selected = set()

        for _ in range(self.k_hops):
            next_frontier = set()
            for ent in frontier:
                for eid in ent_to_eids.get(ent, []):
                    selected.add(eid)
                    e = edges[eid]
                    for neighbor in (e["subj"], e["obj"]):
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier

        return [edges[i]["raw"] for i in sorted(selected)]

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response: str) -> Dict[str, List[Any]]:
        """Parse LLM JSON response into tuple list and entity_class list."""
        empty = {"tuple": [], "entity_class": []}
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            raw_tuples = data.get("tuple", [])
            raw_classes = data.get("entity_class", [])
            return {
                "tuple": [t for t in raw_tuples if isinstance(t, str)] if isinstance(raw_tuples, list) else [],
                "entity_class": raw_classes if isinstance(raw_classes, list) else [],
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return empty

    # ------------------------------------------------------------------
    # Single-row inference
    # ------------------------------------------------------------------

    def infer_single(
        self,
        tuples: List[str],
        entity_pair: List[str],
        ontology: Dict[str, Any],
        prompt_template: FinKGRelationChainInferencePrompt,
        llm_serving: LLMServingABC,
    ) -> Dict[str, List[Any]]:
        """Run inference for one row of tuples."""
        empty = {"tuple": [], "entity_class": []}

        if not isinstance(tuples, list) or not tuples:
            return empty

        e1, e2 = entity_pair

        related = self._find_related_tuples(tuples, e1, e2)
        if not related:
            return empty

        system_prompt = prompt_template.build_system_prompt(ontology)
        user_prompt = prompt_template.build_prompt(
            entity1=e1, entity2=e2, tuples=related,
        )

        responses = llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        if not responses or not responses[0]:
            self.logger.warning("LLM returned empty response")
            return empty

        return self._parse_llm_response(responses[0])


# ======================================================================
# Main operator — thin shell over the engine
# ======================================================================

@OPERATOR_REGISTRY.register()
class FinKGRelationChainInference(OperatorABC):
    """
    LLM-based multi-hop relation chain inference for Financial KG.

    Typical use cases:
      - Ownership penetration  (A→B→C  ⇒ A ultimate_controller_of C)
      - Guarantee chain risk   (A→B→C  ⇒ A indirect_guarantee_risk_for C)
      - Related-party detection (A→X, B→X ⇒ A related_party_of B)
      - Cross-relation impact  (A-lends→B-defaults→C ⇒ A loan_asset_affected_by C)
    """

    def __init__(self, llm_serving: LLMServingABC, lang: str = "en", seed: int = 0):
        self.llm_serving = llm_serving
        self.lang = lang
        self.seed = seed
        self.logger = get_logger()
        self.prompt_template = FinKGRelationChainInferencePrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGRelationChainInference 用于进行金融知识图谱多跳关系链推理。",
                "输入: entity_pair + tuple + ontology; 输出: inferred_tuple + entity_class",
            )
        return (
            "FinKGRelationChainInference is used to perform multi-hop relation-chain reasoning on Financial KG.",
            "Input: entity_pair + tuple + ontology; Output: inferred_tuple + entity_class",
        )

    def _validate_dataframe(self, dataframe, input_key: str):
        if input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: '{input_key}'")

    def run(
        self,
        storage: DataFlowStorage = None,
        entity_pair: List[str] = ["JPMorgan", "Shell Plc"],
        input_key_tuple: str = "tuple",
        input_key_meta: str = "finkg_ontology",
        output_key: str = "inferred_tuple",
        evidence_key: str = "entity_class",
        k_hops: int = 2,
        max_hops: int = 5,
    ) -> List[str]:
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe, input_key_tuple)

        if entity_pair is None or len(entity_pair) != 2:
            raise ValueError("entity_pair must be a list of 2 entities, e.g. ['A','C']")

        tuples_list = dataframe[input_key_tuple].tolist()

        # Read ontology from cached meta file
        storage_meta = FileStorage(first_entry_file_name="", cache_type="json")
        ontology_df = storage_meta.read(
            file_path=f"./.cache/api/{input_key_meta}.json",
            output_type="dataframe",
        )
        row = ontology_df.iloc[0]
        ontology = {
            "entity_type": row["entity_type"],
            "relation_type": row["relation_type"],
            "attribute_type": row.get("attribute_type", {}),
        }

        engine = FinKGRelationChainInferenceEngine(
            k_hops=min(k_hops, max_hops),
            logger=self.logger,
        )

        inferred_rows: List[List[str]] = []
        evidence_rows: List[List[Any]] = []

        for tuples in tqdm(tuples_list, desc="Infer FinKG relation chain"):
            result = engine.infer_single(
                tuples=tuples,
                entity_pair=entity_pair,
                ontology=ontology,
                prompt_template=self.prompt_template,
                llm_serving=self.llm_serving,
            )
            inferred_rows.append(result["tuple"])
            evidence_rows.append(result["entity_class"])

        dataframe[output_key] = inferred_rows
        dataframe[evidence_key] = evidence_rows

        output_file = storage.write(dataframe)
        self.logger.info(f"Inference results saved to {output_file}")

        return [output_key, evidence_key]
