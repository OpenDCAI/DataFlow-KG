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

    Handles three merge scenarios:
        1. Relational + Relational -> deduplicate and detect ambiguous relations
        2. Attribute + Attribute -> deduplicate and detect ambiguous attribute values
        3. Relational + Attribute (or vice versa) -> direct merge, ambiguous empty
    """

    def __init__(self):
        pass

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTripleMerger 用于合并两个知识图谱或属性三元组集合。",
                "支持关系三元组合并、属性三元组合并及混合合并，并自动检测歧义。",
                "输入列 triples_kg1、triples_kg2 为两个图谱的三元组列表，entity_alignment 为实体对齐结果列表；输出列 merged_triples 为合并后的三元组字典（含 unambiguous 和 ambiguous 两个键）。"
            )
        else:
            return (
                "KGTripleMerger merges two KGs or two sets of attribute triples into one.",
                "Supports relational, attribute, and mixed merges with ambiguity detection.",
                "Takes triples_kg1 and triples_kg2 (List[str]) and entity_alignment (List[Dict]) as inputs, and outputs merged_triples (Dict with 'unambiguous' and 'ambiguous' triple lists)."
            )

    # ---------------- Relational Triple Merge ----------------
    @staticmethod
    def _merge_relational_triples(
        triples_kg1: List[str],
        triples_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:

        pair2rels = defaultdict(set)

        def parse_rel_triple(t: str):
            m = re.match(r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)", t)
            if not m:
                return None
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

        def map_entity(e: str) -> str:
            for pair in entity_alignment:
                if e == pair["entity_kg2"]:
                    return pair["entity_kg1"]
            return e

        # Collect KG1 triples
        for t in triples_kg1:
            parsed = parse_rel_triple(t)
            if not parsed:
                continue
            s, o, r = parsed
            key = tuple(sorted([s, o]))
            pair2rels[key].add(r)

        # Collect KG2 triples (with alignment)
        for t in triples_kg2:
            parsed = parse_rel_triple(t)
            if not parsed:
                continue
            s, o, r = parsed
            s = map_entity(s)
            o = map_entity(o)
            key = tuple(sorted([s, o]))
            pair2rels[key].add(r)

        unambiguous, ambiguous = [], []
        for (e1, e2), rels in pair2rels.items():
            if len(rels) == 1:
                rel = next(iter(rels))
                unambiguous.append(f"<subj> {e1} <obj> {e2} <rel> {rel}")
            else:
                rel_str = " | ".join(sorted(rels))
                ambiguous.append(f"<subj> {e1} <obj> {e2} <rel> {rel_str}")

        return {"unambiguous": unambiguous, "ambiguous": ambiguous}

    # ---------------- Attribute Triple Merge ----------------
    @staticmethod
    def _merge_attribute_triples(
        triples_kg1: List[str],
        triples_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:

        attr_dict = defaultdict(lambda: defaultdict(set))

        for triple in triples_kg1 + triples_kg2:
            if not triple.startswith("<entity>"):
                continue
            m = re.match(r"<entity>\s*(.*?)\s*<attribute>\s*(.*?)\s*<value>\s*(.*)", triple)
            if not m:
                continue
            ent, attr, val = m.groups()
            ent_mapped = ent
            for pair in entity_alignment:
                if ent == pair["entity_kg2"]:
                    ent_mapped = pair["entity_kg1"]
                    break
            attr_dict[ent_mapped][attr.strip()].add(val.strip())

        unambiguous, ambiguous = [], []
        for ent, attrs in attr_dict.items():
            for attr, vals in attrs.items():
                if len(vals) == 1:
                    val = next(iter(vals))
                    unambiguous.append(f"<entity> {ent} <attribute> {attr} <value> {val}")
                else:
                    val_str = " | ".join(sorted(vals))
                    ambiguous.append(f"<entity> {ent} <attribute> {attr} <value> {val_str}")

        return {"unambiguous": unambiguous, "ambiguous": ambiguous}

    # ---------------- Mixed Triple Merge ----------------
    @staticmethod
    def _merge_mixed_triples(
        triples_kg1: List[str],
        triples_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:

        merged = []

        def map_entity(e: str) -> str:
            for pair in entity_alignment:
                if e == pair["entity_kg2"]:
                    return pair["entity_kg1"]
            return e

        for t in triples_kg1 + triples_kg2:
            if t.startswith("<subj>"):
                m = re.match(r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)", t)
                if m:
                    s, o, r = m.groups()
                    s = map_entity(s.strip())
                    o = map_entity(o.strip())
                    merged.append(f"<subj> {s} <obj> {o} <rel> {r.strip()}")
                else:
                    merged.append(t)
            elif t.startswith("<entity>"):
                m = re.match(r"<entity>\s*(.*?)\s*<attribute>\s*(.*?)\s*<value>\s*(.*)", t)
                if m:
                    e, a, v = m.groups()
                    e = map_entity(e.strip())
                    merged.append(f"<entity> {e} <attribute> {a.strip()} <value> {v.strip()}")
                else:
                    merged.append(t)
            else:
                merged.append(t)

        return {"unambiguous": merged, "ambiguous": []}

    # ---------------- Run ----------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_kg1: str = "triples_kg1",
        input_key_kg2: str = "triples_kg2",
        input_key_alignment: str = "entity_alignment",
        output_key: str = "merged_triples"
    ) -> List[str]:

        df = storage.read("dataframe")
        triples_kg1 = df[input_key_kg1].tolist()[0]
        triples_kg2 = df[input_key_kg2].tolist()[0]
        alignment = df[input_key_alignment].tolist()[0]

        if not triples_kg1 or not triples_kg2:
            return []

        first1 = triples_kg1[0].strip()
        first2 = triples_kg2[0].strip()

        type1 = "rel" if first1.startswith("<subj>") else "attr"
        type2 = "rel" if first2.startswith("<subj>") else "attr"

        if type1 == "rel" and type2 == "rel":
            merged = self._merge_relational_triples(triples_kg1, triples_kg2, alignment)
        elif type1 == "attr" and type2 == "attr":
            merged = self._merge_attribute_triples(triples_kg1, triples_kg2, alignment)
        else:
            merged = self._merge_mixed_triples(triples_kg1, triples_kg2, alignment)

        df[output_key] = [merged]
        storage.write(df)
        return [output_key]