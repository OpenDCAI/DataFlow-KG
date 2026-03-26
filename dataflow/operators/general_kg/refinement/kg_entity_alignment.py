from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from typing import List, Dict
from fuzzywuzzy import fuzz
import pandas as pd
import re


@OPERATOR_REGISTRY.register()
class KGGraphEntityAligner(OperatorABC):
    """
    Entity alignment operator for matching entities between two KGs given as triple lists.
    """

    def __init__(self, top_k: int = 5, threshold: int = 70):
        """
        Args:
            top_k (int): Maximum number of candidate entities per source entity
            threshold (int): Minimum similarity score to consider a candidate
        """
        self.top_k = top_k
        self.threshold = threshold

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGGraphEntityAligner 用于对齐两个知识图谱中的实体。",
                "通过模糊字符串匹配生成候选对齐，并选取相似度最高的匹配。",
                "输入列 triples_kg1 和 triples_kg2 为两个图谱的三元组字符串列表，输出列 entity_alignment 为实体对齐结果列表，每项包含 entity_kg1、entity_kg2 和 similarity 字段。"
            )
        else:
            return (
                "KGGraphEntityAligner aligns entities between two KG triple lists.",
                "Generates candidates via fuzzy string matching and selects the highest-similarity match.",
                "Takes triples_kg1 and triples_kg2 (List[str]) as inputs and outputs entity_alignment (List[Dict] with entity_kg1, entity_kg2, and similarity fields)."
            )

    # ---------------- Entity Extraction ----------------
    def _extract_entities(self, tuples: List[str]) -> List[str]:
        """
        Extract unique entities from n-ary tuples.
        Strategy:
            - <subj> / <obj> for relational triples
            - <entity> for attribute-like tuples
            - Generic: consider any token after known entity keywords
        """
        entities = set()
        entity_keys = ["<subj>", "<obj>", "<entity>"]

        for tup in tuples:
            for key in entity_keys:
                pattern = re.compile(rf"{key}\s*([^<]+)")
                matches = pattern.findall(tup)
                for m in matches:
                    if m.strip():
                        entities.add(m.strip())
        return list(entities)

    # ---------------- Candidate Generation ----------------
    def _generate_candidates(
        self,
        entities_kg1: List[str],
        entities_kg2: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Generate candidate entity pairs using fuzzy string matching.
        """
        candidate_dict = {}
        for e1 in entities_kg1:
            scores = []
            for e2 in entities_kg2:
                score = fuzz.ratio(e1.lower(), e2.lower())  # APPLE / APPLE Inc.
                if score >= self.threshold:
                    scores.append({"entity_kg2": e2, "similarity": score})
            # Keep top_k candidates
            scores = sorted(scores, key=lambda x: x["similarity"], reverse=True)[:self.top_k]
            candidate_dict[e1] = scores
        return candidate_dict

    # ---------------- Alignment ----------------
    def _align_entities(self, candidate_dict: Dict[str, List[Dict]]) -> List[Dict[str, str]]:
        """
        Align entities by picking the highest similarity candidate.
        """
        alignments = []
        for e1, candidates in candidate_dict.items():
            if not candidates:
                continue
            best_match = max(candidates, key=lambda x: x["similarity"])
            alignments.append({
                "entity_kg1": e1,
                "entity_kg2": best_match["entity_kg2"],
                "similarity": best_match["similarity"]
            })
        return alignments

    # ---------------- Run ----------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_kg1: str = "triples_kg1",
        input_key_kg2: str = "triples_kg2",
        output_key: str = "entity_alignment"
    ):
        """
        Run the entity alignment on two KG triple lists.
        """
        df = storage.read("dataframe")
        triples_kg1 = df[input_key_kg1].tolist()[0]  # assuming single row
        triples_kg2 = df[input_key_kg2].tolist()[0]

        # Step 1: extract entities
        entities_kg1 = self._extract_entities(triples_kg1)
        entities_kg2 = self._extract_entities(triples_kg2)

        # Step 2: generate candidates
        candidate_dict = self._generate_candidates(entities_kg1, entities_kg2)

        # Step 3: align entities
        alignments = self._align_entities(candidate_dict)

        df[output_key] = [alignments]
        storage.write(df)

        return [output_key]
