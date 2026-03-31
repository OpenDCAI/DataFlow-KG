import re
from collections import defaultdict, deque
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KGReasoningPathSearch(OperatorABC):
    """
    Find all multi-hop paths connecting target entities in a KG.

    Input columns:
        - triplet: List[str]
        - target_entity: str or List[List[str]]

    Output columns:
        - mpath: List[List[List[str]]]   # paths grouped by input entity pairs
    """

    def __init__(self, max_hop: int = 10):
        self.max_hop = max_hop
        self.logger = get_logger()
        self.rel_pattern = re.compile(
            r"<subj>\s*(.+?)\s*<obj>\s*(.+?)\s*<rel>\s*(.+?)$"
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGReasoningPathSearch 用于搜索目标实体对之间的多跳路径。",
                "输入: triplet + target_entity; 输出: mpath",
            )
        return (
            "KGReasoningPathSearch is used to search multi-hop paths between target entity pairs.",
            "Input: triplet + target_entity; Output: mpath",
        )

    # ----------------------------
    # Target normalization
    # ----------------------------
    def _normalize_targets(self, target_raw) -> List[str]:
        """
        Normalize target_entity into List[str]
        """
        targets = []
        if isinstance(target_raw, str):
            targets = [t.strip() for t in target_raw.split(",") if t.strip()]
        elif isinstance(target_raw, list):
            for item in target_raw:
                if isinstance(item, str):
                    targets.extend([t.strip() for t in item.split(",") if t.strip()])
        return list(dict.fromkeys(targets))

    def _normalize_target_pair(self, raw) -> List[str]:
        """
        Normalize a single target pair:
        ["Henry, Berlin"] or "Henry, Berlin" -> ["Henry", "Berlin"]
        """
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if isinstance(raw, str):
            return [t.strip() for t in raw.split(",") if t.strip()]
        return []

    # ----------------------------
    # Triplet parsing
    # ----------------------------
    def _parse_triplet(self, t: str) -> Tuple[str, str, str]:
        m = self.rel_pattern.search(t)
        if not m:
            return None, None, None
        return m.group(1).strip(), m.group(3).strip(), m.group(2).strip()

    # ----------------------------
    # Build undirected graph
    # ----------------------------
    def _build_graph(self, triplets: List[str]):
        adj = defaultdict(list)
        for t in triplets:
            h, r, o = self._parse_triplet(t)
            if h is None:
                continue
            adj[h].append((o, t))
            adj[o].append((h, t))  # undirected
        return adj

    # ----------------------------
    # BFS path search (all paths)
    # ----------------------------
    def _find_paths_between(self, adj, src: str, tgt: str) -> List[List[str]]:
        """
        Find all simple paths from src to tgt within max_hop
        """
        paths = []
        queue = deque()
        queue.append((src, [], {src}))  # node, path(triples), visited set

        while queue:
            cur, cur_path, visited = queue.popleft()

            if len(cur_path) > self.max_hop:
                continue

            if cur == tgt and cur_path:
                paths.append(cur_path)
                continue

            for nxt, triple in adj.get(cur, []):
                if nxt in visited:
                    continue
                queue.append(
                    (nxt, cur_path + [triple], visited | {nxt})
                )

        return paths

    # ----------------------------
    # Validation
    # ----------------------------
    def _validate_dataframe(self, df: pd.DataFrame):
        for col in ["triplet", "target_entity"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if "mpath" in df.columns:
            raise ValueError("Output column already exists: mpath")

    # ----------------------------
    # Run
    # ----------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "mpath",
    ) -> List[str]:

        if storage is None:
            raise ValueError("storage cannot be None")

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        all_paths = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="KG multi-hop path reasoning"):
            triplets = row[input_key]
            target_raw = row["target_entity"]

            if not isinstance(triplets, list):
                all_paths.append([])
                continue

            adj = self._build_graph(triplets)

            row_paths = []  # paths grouped by entity pair

            # -------- CASE 1: 新格式 List[List[str]] --------
            if isinstance(target_raw, list) and target_raw and isinstance(target_raw[0], list):
                for pair_raw in target_raw:
                    pair = self._normalize_target_pair(pair_raw)
                    if len(pair) != 2:
                        row_paths.append([])
                        continue
                    src, tgt = pair
                    paths = self._find_paths_between(adj, src, tgt)
                    row_paths.append(paths)

            # -------- CASE 2: 旧格式兼容 --------
            else:
                targets = self._normalize_targets(target_raw)
                if len(targets) >= 2:
                    for i in range(len(targets)):
                        for j in range(i + 1, len(targets)):
                            src, tgt = targets[i], targets[j]
                            paths = self._find_paths_between(adj, src, tgt)
                            row_paths.append(paths)

            all_paths.append(row_paths)

        df[output_key] = all_paths
        output_file = storage.write(df)
        self.logger.info(f"Multi-hop paths saved to {output_file}")

        return [output_key]
