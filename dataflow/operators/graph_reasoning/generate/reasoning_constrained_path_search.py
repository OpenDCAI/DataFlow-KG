"""
====================================
DataFlow-KG: KGReasoningConstrainedPathSearch
====================================
"""

import re
from collections import defaultdict
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class KGReasoningConstrainedPathSearch(OperatorABC):
    """
    Constrained multi-hop path search in Knowledge Graph.

    Input:
        - triplet: List[str]
        - target_entity: List[List[str]] or str / List[str] (backward compatible)

    Output:
        - cons_mpath: List[List[List[str]]]
          每个元素对应一个实体对的所有路径
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        max_hop: int = 3,
        must_pass_entities: List[str] = None,
        allowed_relations: List[str] = None,
        required_entity_types: List[str] = None,
        entity_type_map: Dict[str, str] = None,
    ):
        self.max_hop = max_hop
        self.must_pass_entities = set(must_pass_entities or [])
        self.allowed_relations = set(allowed_relations or [])
        self.required_entity_types = set(required_entity_types or [])
        self.entity_type_map = entity_type_map or {}

        self.logger = get_logger()
        self.rel_pattern = re.compile(
            r"<subj>\s*(.+?)\s*<obj>\s*(.+?)\s*<rel>\s*(.+?)$"
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGReasoningConstrainedPathSearch 用于按约束搜索知识图谱多跳路径。",
                "输入: triplet + target_entity + constraints; 输出: cons_mpath",
            )
        return (
            "KGReasoningConstrainedPathSearch is used to search constrained multi-hop paths in a knowledge graph.",
            "Input: triplet + target_entity + constraints; Output: cons_mpath",
        )

    # --------------------------------------------------
    # Target normalization
    # --------------------------------------------------
    def _normalize_targets(self, target_raw) -> List[str]:
        targets = []
        if isinstance(target_raw, str):
            targets = [t.strip() for t in target_raw.split(",") if t.strip()]
        elif isinstance(target_raw, list):
            for item in target_raw:
                if isinstance(item, str):
                    targets.extend([t.strip() for t in item.split(",") if t.strip()])
        return list(dict.fromkeys(targets))

    def _normalize_target_pair(self, raw) -> List[str]:
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if isinstance(raw, str):
            return [t.strip() for t in raw.split(",") if t.strip()]
        return []

    # --------------------------------------------------
    # Graph construction
    # --------------------------------------------------
    def _build_graph(self, triples: List[str]):
        graph = defaultdict(list)
        for t in triples:
            m = self.rel_pattern.search(t)
            if not m:
                continue
            subj, obj, rel = m.groups()
            if self.allowed_relations and rel not in self.allowed_relations:
                continue
            graph[subj].append((obj, rel, t))
        return graph

    # --------------------------------------------------
    # Constraint checking
    # --------------------------------------------------
    def _check_constraints(self, path_entities: List[str]) -> bool:
        if self.must_pass_entities and not self.must_pass_entities.issubset(set(path_entities)):
            return False
        if self.required_entity_types:
            types_in_path = {self.entity_type_map.get(e) for e in path_entities}
            if not self.required_entity_types.intersection(types_in_path):
                return False
        return True

    # --------------------------------------------------
    # DFS path search (找到所有路径)
    # --------------------------------------------------
    def _dfs_paths(self, graph, start: str, target: str):
        results = []
        stack = [(start, [], [start])]  # node, path(triples), entities

        while stack:
            node, path, entities = stack.pop()
            if len(path) > self.max_hop:
                continue
            if node == target and path:
                if self._check_constraints(entities):
                    results.append(path)
                continue
            for nxt, rel, triple_str in graph.get(node, []):
                if nxt in entities:  # 单条路径内避免重复节点
                    continue
                stack.append((nxt, path + [triple_str], entities + [nxt]))
        return results

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage,
        triplet_key: str = "triplet",
        target_key: str = "target_entity",
        output_key: str = "cons_mpath",
    ):
        df = storage.read("dataframe")
        all_results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Constrained path search"):
            triples = row[triplet_key]
            graph = self._build_graph(triples)

            target_raw = row[target_key]
            paths_for_row = []  # 每个实体对的路径列表

            # -------- CASE 1: 新格式 List[List[str]] --------
            if isinstance(target_raw, list) and target_raw and isinstance(target_raw[0], list):
                for pair_raw in target_raw:
                    pair = self._normalize_target_pair(pair_raw)
                    if len(pair) != 2:
                        continue
                    paths_for_row.append(self._dfs_paths(graph, pair[0], pair[1]))

            # -------- CASE 2: 旧格式兼容 --------
            else:
                targets = self._normalize_targets(target_raw)
                if len(targets) >= 2:
                    for i in range(len(targets)):
                        for j in range(i + 1, len(targets)):
                            paths_for_row.append(self._dfs_paths(graph, targets[i], targets[j]))

            all_results.append(paths_for_row)

        df[output_key] = all_results
        output_file = storage.write(df)
        self.logger.info(f"KGConstrainedPathSearch results saved to {output_file}")
        return [output_key]
