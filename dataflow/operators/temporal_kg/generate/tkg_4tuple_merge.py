from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from typing import List, Dict
import re
from collections import defaultdict

@OPERATOR_REGISTRY.register()
class TKGTupleMerger(OperatorABC):
    """
    Merge two KGs or attribute sets of quadruples into a unified KG.
    Supports:
    - Relational quadruples: <subj> ... <obj> ... <rel> ... <time> ...
    - Attribute quadruples: <entity> ... <attribute> ... <value> ... <time> ...

    Ambiguity can occur in four cases:
    1. R4 relation conflict: same subj-obj-time, different relation
    2. R4 time conflict: same subj-rel-obj, different time
    3. A4 value conflict: same entity-attribute-time, different value
    4. A4 time conflict: same entity-attribute-value, different time
    """

    def __init__(self):
        pass

    # ---------------- Relational quadruple merge ----------------
    @staticmethod
    def _merge_relational_quads(
        quads_kg1: List[str],
        quads_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Merge R4 quadruples with entity alignment.
        Ambiguous quadruples are joined by '｜'.
        """
        key2rel = defaultdict(set)
        key2time = defaultdict(set)

        def parse_r4(q: str):
            m = re.match(r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*?)\s*<time>\s*(.*)", q)
            if not m:
                return None
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()

        def map_entity(e: str):
            for pair in entity_alignment:
                if e == pair["entity_kg2"]:
                    return pair["entity_kg1"]
            return e

        # Collect KG1
        for q in quads_kg1:
            parsed = parse_r4(q)
            if not parsed:
                continue
            s, o, r, t = parsed
            key2rel[(s, o, t)].add(r)
            key2time[(s, r, o)].add(t)

        # Collect KG2 with alignment
        for q in quads_kg2:
            parsed = parse_r4(q)
            if not parsed:
                continue
            s, o, r, t = parsed
            s = map_entity(s)
            o = map_entity(o)
            key2rel[(s, o, t)].add(r)
            key2time[(s, r, o)].add(t)

        unambiguous = []
        ambiguous = []

        # Handle relation and time conflicts
        for (s, o, t), rels in key2rel.items():
            times_for_this_rel = key2time.get((s, next(iter(rels)), o), set())
            if len(rels) == 1 and len(times_for_this_rel) == 1:
                r = next(iter(rels))
                unambiguous.append(f"<subj> {s} <obj> {o} <rel> {r} <time> {t}")
            else:
                # 拼接所有冲突组合
                quads_list = []
                for r in rels:
                    t_for_this_r = key2time.get((s, r, o), set())
                    for time in t_for_this_r or [t]:
                        quads_list.append(f"<subj> {s} <obj> {o} <rel> {r} <time> {time}")
                ambiguous.append(" ｜ ".join(sorted(quads_list)))

        return {"unambiguous": unambiguous, "ambiguous": ambiguous}

    # ---------------- Attribute quadruple merge ----------------
    @staticmethod
    def _merge_attribute_quads(
        quads_kg1: List[str],
        quads_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Merge A4 quadruples with entity alignment.
        Ambiguous quadruples are joined by '｜'.
        """
        key2vals = defaultdict(set)
        key2times = defaultdict(set)

        def parse_a4(q: str):
            m = re.match(r"<entity>\s*(.*?)\s*<attribute>\s*(.*?)\s*<value>\s*(.*?)\s*<time>\s*(.*)", q)
            if not m:
                return None
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()

        def map_entity(e: str):
            for pair in entity_alignment:
                if e == pair["entity_kg2"]:
                    return pair["entity_kg1"]
            return e

        for q in quads_kg1 + quads_kg2:
            parsed = parse_a4(q)
            if not parsed:
                continue
            ent, attr, val, t = parsed
            ent = map_entity(ent)
            key2vals[(ent, attr, t)].add(val)
            key2times[(ent, attr, val)].add(t)

        unambiguous = []
        ambiguous = []

        # Handle value and time conflicts
        for (ent, attr, t), vals in key2vals.items():
            times_for_vals = {val: key2times.get((ent, attr, val), {t}) for val in vals}
            if len(vals) == 1 and len(next(iter(times_for_vals.values()))) == 1:
                val = next(iter(vals))
                unambiguous.append(f"<entity> {ent} <attribute> {attr} <value> {val} <time> {t}")
            else:
                quads_list = []
                for val in vals:
                    for time in times_for_vals[val]:
                        quads_list.append(f"<entity> {ent} <attribute> {attr} <value> {val} <time> {time}")
                ambiguous.append(" ｜ ".join(sorted(quads_list)))

        return {"unambiguous": unambiguous, "ambiguous": ambiguous}

    # ---------------- Run ----------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_kg1: str = "triples_kg1",
        input_key_kg2: str = "triples_kg2",
        input_key_alignment: str = "entity_alignment",
        output_key: str = "merged_quads"
    ):
        """
        Merge KG1 and KG2 quadruples (relational or attribute)
        """
        df = storage.read("dataframe")
        quads_kg1 = df[input_key_kg1].tolist()[0]
        quads_kg2 = df[input_key_kg2].tolist()[0]
        alignment = df[input_key_alignment].tolist()[0]

        if not quads_kg1:
            return []

        first = quads_kg1[0].strip()
        if first.startswith("<subj>"):
            merged = self._merge_relational_quads(quads_kg1, quads_kg2, alignment)
        elif first.startswith("<entity>"):
            merged = self._merge_attribute_quads(quads_kg1, quads_kg2, alignment)
        else:
            raise ValueError("Unknown quadruple type detected.")

        df[output_key] = [merged]
        storage.write(df)
        return [output_key]