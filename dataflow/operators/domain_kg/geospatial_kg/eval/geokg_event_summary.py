import json
import re
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


class GeoKGTupleAttributeFrequencyEvaluator(OperatorABC):
    """
    Evaluate the frequency of attributes in KG tuples.

    Input:
        - tuple: List[str] in format:
          "<subj> X <obj> Y <rel> Z <Time> ... <Location> ... <Value> ..."

    Output:
        - attribute_counts: Dict[str, int]
        - attribute_frequencies: Dict[str, float]
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        # 匹配属性标签：<Time>, <Location>, <Value>, <Capacity> 等
        self.attr_pattern = re.compile(r"<([A-Za-z0-9_]+)>\s*[^<]+")

    # ============================================================
    # Parse attributes
    # ============================================================
    def _extract_attributes(self, tuple_str: str) -> List[str]:
        """
        Extract attribute labels from a tuple.
        Example:
        "<subj> Elon Musk <obj> Announcement <rel> MadeAt <Time> May 15, 2025 <Location> Berlin"
        -> ["Time", "Location"]
        """
        if not isinstance(tuple_str, str):
            return []

        return [m.group(1) for m in self.attr_pattern.finditer(tuple_str)]

    # ============================================================
    # Batch processing
    # ============================================================
    def process_batch(self, dataframe_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        attr_counter = defaultdict(int)
        total_tuples = 0

        for row in tqdm(dataframe_subset, desc="Counting Attributes"):

            tuples = row.get("tuple", [])

            if isinstance(tuples, str):
                try:
                    tuples = json.loads(tuples)
                except:
                    tuples = []

            if not isinstance(tuples, list):
                tuples = []

            for t in tuples:
                total_tuples += 1
                attrs = self._extract_attributes(t)
                for a in attrs:
                    attr_counter[a] += 1

        # 计算频率
        attr_freq = {k: v / total_tuples for k, v in attr_counter.items()} if total_tuples > 0 else {}

        return {
            self.output_key: dict(attr_counter),
            self.output_key_meta: attr_freq
        }

    # ============================================================
    # Run
    # ============================================================
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

        # 组织记录
        records = []
        for _, r in df.iterrows():
            records.append({
                "tuple": r.get(input_key, [])
            })

        output = self.process_batch(records)

        # 保存回 storage
        out_file = pd.DataFrame([{
            output_key: output[output_key],
            output_key_meta: output[output_key_meta]
        }])
        out_file = storage.write(out_file)
        self.logger.info(f"Attribute frequency evaluation complete. Saved to {out_file}")

        return [output_key, output_key_meta]