import json
import re
from typing import List, Dict, Any
from collections import defaultdict

from tqdm import tqdm
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


class HRKGTupleAttributeFrequencyEvaluator(OperatorABC):
    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        self.attr_pattern = re.compile(r"<([A-Za-z0-9_]+)>\s*[^<]+")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "HRKGTupleAttributeFrequencyEvaluator 用于统计知识图谱 tuple 字段中各类属性标签的出现次数与频率分布。",
                "输入: 包含 tuple 字段的数据，tuple 可以是字符串形式的 JSON 列表，也可以是 Python 列表；列表中的每个元素通常是形如 "
                "\"<subj> ... <obj> ... <rel> ... <Time> ... <Location> ... <Value> ...\" 的三元组/事件字符串。"
                "算子会从每条 tuple 中抽取尖括号属性标签，如 Time、Location、Value、Capacity 等，并在整个数据集范围内做聚合统计。"
                "输出: attribute_counts 和 attribute_frequencies。"
                "其中 attribute_counts 是属性标签到出现次数的映射字典，如 {\"Time\": 120, \"Location\": 85}；"
                "attribute_frequencies 是属性标签到相对频率的映射字典，频率按“出现该属性的次数 / tuple 总数”计算。",
            )
        return (
            "HRKGTupleAttributeFrequencyEvaluator is used to compute the occurrence counts and frequency distribution of attribute labels in KG tuples.",
            "Input: a dataset containing a tuple field, where tuple can be either a JSON-encoded string list or a Python list. "
            "Each element in the list is typically a tuple/event string in a format such as "
            "\"<subj> ... <obj> ... <rel> ... <Time> ... <Location> ... <Value> ...\". "
            "The operator extracts attribute tags enclosed in angle brackets, such as Time, Location, Value, Capacity, etc., "
            "and aggregates them over the whole dataset. "
            "Output: attribute_counts and attribute_frequencies. "
            "attribute_counts is a dictionary mapping each attribute label to its total count, e.g. {\"Time\": 120, \"Location\": 85}; "
            "attribute_frequencies is a dictionary mapping each attribute label to its relative frequency, "
            "computed as attribute occurrence count divided by the total number of tuples.",
        )

    def _extract_attributes(self, tuple_str: str) -> List[str]:
        if not isinstance(tuple_str, str):
            return []
        return [m.group(1) for m in self.attr_pattern.finditer(tuple_str)]

    def process_batch(self, dataframe_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        attr_counter = defaultdict(int)
        total_tuples = 0

        for row in tqdm(dataframe_subset, desc="Counting Attributes"):
            tuples = row.get("tuple", [])

            if isinstance(tuples, str):
                try:
                    tuples = json.loads(tuples)
                except Exception:
                    tuples = []

            if not isinstance(tuples, list):
                tuples = []

            for t in tuples:
                total_tuples += 1
                attrs = self._extract_attributes(t)
                for a in attrs:
                    attr_counter[a] += 1

        attr_freq = {k: v / total_tuples for k, v in attr_counter.items()} if total_tuples > 0 else {}

        return {
            self.output_key: dict(attr_counter),
            self.output_key_meta: attr_freq
        }

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key: str = "attribute_counts",
        output_key_meta: str = "attribute_frequencies",
    ) -> List[str]:
        self.input_key, self.output_key, self.output_key_meta = input_key, output_key, output_key_meta
        if storage is None:
            raise ValueError("Storage is required.")

        df = storage.read("dataframe")
        self.logger.info(f"Starting Attribute Frequency Eval on {len(df)} records.")

        records = []
        for _, r in df.iterrows():
            records.append({
                "tuple": r.get(input_key, [])
            })

        output = self.process_batch(records)

        out_file = pd.DataFrame([{
            output_key: output[output_key],
            output_key_meta: output[output_key_meta]
        }])
        out_file = storage.write(out_file)
        self.logger.info(f"Attribute frequency evaluation complete. Saved to {out_file}")

        return [output_key, output_key_meta]