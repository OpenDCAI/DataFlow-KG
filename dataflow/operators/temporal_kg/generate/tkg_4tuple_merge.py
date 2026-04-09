from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from typing import List, Dict
import re
from collections import defaultdict


@OPERATOR_REGISTRY.register()
class TKGTupleMerger(OperatorABC):
    def __init__(self):
        pass

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "TKGTupleMerger 用于将两个时序知识图谱或属性四元组集合合并为统一结果，并区分无歧义合并结果与存在冲突的歧义结果。",
                "输入: 数据表中需要包含两个四元组列表字段以及一个实体对齐字段。"
                "其中 input_key_kg1 和 input_key_kg2 分别表示待合并的两个知识图谱四元组列表，"
                "支持两类输入格式：关系四元组 "
                "\"<subj> ... <obj> ... <rel> ... <time> ...\"，"
                "以及属性四元组 "
                "\"<entity> ... <attribute> ... <value> ... <time> ...\"。"
                "input_key_alignment 对应实体对齐结果，通常为字典列表，每个字典至少包含 entity_kg1 和 entity_kg2，用于将 KG2 中的实体映射到 KG1。"
                "算子会自动识别输入是关系四元组还是属性四元组，并在合并过程中处理关系冲突、时间冲突、属性值冲突等情况。"
                "输出: merged_quads。该字段为一个字典，通常包含 unambiguous 和 ambiguous 两个键；"
                "unambiguous 表示无歧义的合并结果列表，ambiguous 表示存在冲突的候选四元组组合列表。"
                "对于歧义结果，多个候选四元组会使用全角分隔符 '｜' 连接成一个字符串。",
            )
        return (
            "TKGTupleMerger is used to merge two temporal knowledge graphs or attribute-quadruple sets into a unified result while separating unambiguous merged results from ambiguous conflict cases.",
            "Input: the dataframe must contain two quadruple-list fields and one entity-alignment field. "
            "input_key_kg1 and input_key_kg2 represent the two KG quadruple lists to be merged. "
            "Two input formats are supported: relational quadruples in the form "
            "\"<subj> ... <obj> ... <rel> ... <time> ...\", "
            "and attribute quadruples in the form "
            "\"<entity> ... <attribute> ... <value> ... <time> ...\". "
            "input_key_alignment stores entity alignment results, usually as a list of dictionaries, where each dictionary contains at least entity_kg1 and entity_kg2, "
            "so that entities in KG2 can be mapped to KG1 before merging. "
            "The operator automatically determines whether the input contains relational or attribute quadruples, and handles conflicts such as relation conflicts, time conflicts, and value conflicts during merging. "
            "Output: merged_quads. This field is a dictionary that usually contains two keys: unambiguous and ambiguous. "
            "unambiguous stores the merged quadruples without conflicts, while ambiguous stores candidate quadruple combinations with conflicts. "
            "For ambiguous results, multiple candidate quadruples are concatenated into a single string using the full-width separator '｜'.",
        )

    @staticmethod
    def _merge_relational_quads(
        quads_kg1: List[str],
        quads_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:
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

        for q in quads_kg1:
            parsed = parse_r4(q)
            if not parsed:
                continue
            s, o, r, t = parsed
            key2rel[(s, o, t)].add(r)
            key2time[(s, r, o)].add(t)

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

        for (s, o, t), rels in key2rel.items():
            times_for_this_rel = key2time.get((s, next(iter(rels)), o), set())
            if len(rels) == 1 and len(times_for_this_rel) == 1:
                r = next(iter(rels))
                unambiguous.append(f"<subj> {s} <obj> {o} <rel> {r} <time> {t}")
            else:
                quads_list = []
                for r in rels:
                    t_for_this_r = key2time.get((s, r, o), set())
                    for time in t_for_this_r or [t]:
                        quads_list.append(f"<subj> {s} <obj> {o} <rel> {r} <time> {time}")
                ambiguous.append(" ｜ ".join(sorted(quads_list)))

        return {"unambiguous": unambiguous, "ambiguous": ambiguous}

    @staticmethod
    def _merge_attribute_quads(
        quads_kg1: List[str],
        quads_kg2: List[str],
        entity_alignment: List[Dict]
    ) -> Dict[str, List[str]]:
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

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_kg1: str = "triples_kg1",
        input_key_kg2: str = "triples_kg2",
        input_key_alignment: str = "entity_alignment",
        output_key: str = "merged_tuples"
    ):
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