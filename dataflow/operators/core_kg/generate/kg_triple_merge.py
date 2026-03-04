"""
====================================
DataFlow-KG: KGTripleMerger
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-28

License:
    MIT License
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from typing import List, Dict
import re
from collections import defaultdict

@OPERATOR_REGISTRY.register()
class KGTripleMerger(OperatorABC):
    """
    Merge two KGs or two sets of attribute triples into a single KG.
    Supports:
    - Relational triples ("<subj> ... <obj> ... <rel> ...")
    - Attribute triples ("<entity> ... <attribute> ... <value> ...")

    For relational triples:
        - Merge KG2 into KG1 using entity_alignment
        - Deduplicate triples

    For attribute triples:
        - Merge KG2 into KG1 using entity_alignment
        - Split into unambiguous and ambiguous triples
    """

    def __init__(self):
        # No complex config needed for now
        pass

    @staticmethod
    def _merge_relational_triples(
        triples_kg1: List[str],
        triples_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Merge relational triples with entity alignment.

        Returns:
            {
                "unambiguous": [...],
                "ambiguous": [...]
            }
        """

        pair2rels = defaultdict(set)

        def parse_rel_triple(t: str):
            m = re.match(
                r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)", t
            )
            if not m:
                return None
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

        def map_entity(e: str) -> str:
            for pair in entity_alignment:
                if e == pair["entity_kg2"]:
                    return pair["entity_kg1"]
            return e

        # ---------- Collect KG1 ----------
        for t in triples_kg1:
            parsed = parse_rel_triple(t)
            if not parsed:
                continue
            s, o, r = parsed
            key = tuple(sorted([s, o]))
            pair2rels[key].add(r)

        # ---------- Collect KG2 (with alignment) ----------
        for t in triples_kg2:
            parsed = parse_rel_triple(t)
            if not parsed:
                continue
            s, o, r = parsed
            s_mapped = map_entity(s)
            o_mapped = map_entity(o)
            key = tuple(sorted([s_mapped, o_mapped]))
            pair2rels[key].add(r)

        unambiguous = []
        ambiguous = []

        for (e1, e2), rels in pair2rels.items():
            if len(rels) == 1:
                rel = next(iter(rels))
                unambiguous.append(
                    f"<subj> {e1} <obj> {e2} <rel> {rel}"
                )
            else:
                rel_str = " | ".join(sorted(rels))
                ambiguous.append(
                    f"<subj> {e1} <obj> {e2} <rel> {rel_str}"
                )

        return {
            "unambiguous": unambiguous,
            "ambiguous": ambiguous,
        }


    # ---------------- Attribute Triple Merge ----------------
    @staticmethod
    def _merge_attribute_triples(
        triples_kg1: List[str],
        triples_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Merge attribute triples using entity alignment.
        Returns:
            {
                "unambiguous": [...],
                "ambiguous": [...]
            }
        """
        # Collect attributes per entity
        attr_dict = defaultdict(lambda: defaultdict(set))

        for triple in triples_kg1 + triples_kg2:
            if not triple.startswith("<entity>"):
                continue
            match = re.match(r"<entity>\s*(.*?)\s*<attribute>\s*(.*?)\s*<value>\s*(.*)", triple)
            if not match:
                continue
            ent, attr, val = match.groups()
            # Map KG2 entities to KG1
            ent_mapped = ent
            for pair in entity_alignment:
                if ent == pair["entity_kg2"]:
                    ent_mapped = pair["entity_kg1"]
                    break
            attr_dict[ent_mapped][attr.strip()].add(val.strip())

        unambiguous = []
        ambiguous = []

        for ent, attrs in attr_dict.items():
            for attr, vals in attrs.items():
                if len(vals) == 1:
                    val = next(iter(vals))
                    unambiguous.append(f"<entity> {ent} <attribute> {attr} <value> {val}")
                else:
                    val_str = " | ".join(sorted(vals))
                    ambiguous.append(f"<entity> {ent} <attribute> {attr} <value> {val_str}")

        return {"unambiguous": unambiguous, "ambiguous": ambiguous}

    # ---------------- Run ----------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_kg1: str = "triples_kg1",
        input_key_kg2: str = "triples_kg2",
        input_key_alignment: str = "entity_alignment",
        output_key_relational: str = "merged_triples",
        output_key_attribute: str = "merged_triples"
    ):
        """
        Merge KG1 and KG2 triples (either relational or attribute) into a unified KG.
        Automatically determines triple type based on first triple of KG1.
        """
        df = storage.read("dataframe")
        triples_kg1 = df[input_key_kg1].tolist()[0]
        triples_kg2 = df[input_key_kg2].tolist()[0]
        alignment = df[input_key_alignment].tolist()[0]

        if not triples_kg1:
            return []

        # Determine type based on first triple
        first = triples_kg1[0].strip()
        if first.startswith("<subj>"):
            merged = self._merge_relational_triples(triples_kg1, triples_kg2, alignment)
            df[output_key_relational] = [merged]
            storage.write(df)
            return [output_key_relational]
        elif first.startswith("<entity>"):
            merged_attrs = self._merge_attribute_triples(triples_kg1, triples_kg2, alignment)
            df[output_key_attribute] = [merged_attrs]
            storage.write(df)
            return [output_key_attribute]
        else:
            raise ValueError("Unknown triple type detected.")