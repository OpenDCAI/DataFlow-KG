"""
====================================
DataFlow-KG:
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

import pandas as pd
import random
import re
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class HRKGEntityBasedSubgraphSampling(OperatorABC):
    """
    Knowledge Graph subgraph sampling operator.
    Supports BFS / Random Walk / Hop-based sampling.
    """
    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        num_q: int = 5
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()


    def _parse_triple(self, triple_str: str) -> tuple[str, str, str]:
        """
        Parse '<subj> ... <obj> ... <rel> ... <attr1> ... <attr2> ...' -> (subject, full_relation_str, object)
        - subject: 紧跟 <subj> 的实体名
        - object: 紧跟 <obj> 的实体名
        - full_relation_str: <rel> + 所有附加属性原文（如 <Time>, <Location>, <Value> 等）
        """

        triple_str = triple_str.strip()

        # 找 <subj>
        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        if not subj_match:
            raise ValueError(f"No <subj> found in triple: {triple_str}")
        subj = subj_match.group(1).strip()

        # 找 <obj>
        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        if not obj_match:
            raise ValueError(f"No <obj> found in triple: {triple_str}")
        obj = obj_match.group(1).strip()

        # 找 <rel> 到字符串末尾（包含后续所有属性）
        rel_match = re.search(r"<rel>\s*(.+)$", triple_str)
        if not rel_match:
            raise ValueError(f"No <rel> found in triple: {triple_str}")
        full_rel = rel_match.group(1).strip()

        # 保留所有 <...> 属性
        # full_rel 已经包含 <Time>, <Location>, <Value>, <Capacity>, <Market>, <Method>, 等全部内容
        return subj, full_rel, obj


    def _collect_entities(self, triple_groups: List[List[str]]) -> List[str]:
        entities = set()
        for group in triple_groups:
            for t in group:
                h, r, tail = self._parse_triple(t)
                entities.add(h)
                entities.add(tail)
        return list(entities)

    # ------------------------------------------------------------------
    # BFS sampling
    # ------------------------------------------------------------------

    def _bfs_sample_subgraph(
        self,
        triple_groups: List[List[str]],
        M: int,
        start_entity: str
    ) -> List[str]:

        triples = []
        for group in triple_groups:
            for t in group:
                triples.append(self._parse_triple(t))

        entity_to_triples = defaultdict(list)
        for idx, (h, r, t) in enumerate(triples):
            entity_to_triples[h].append(idx)
            entity_to_triples[t].append(idx)

        if start_entity not in entity_to_triples:
            return []

        visited_entities = {start_entity}
        visited_triples = set()
        queue = deque([start_entity])

        while queue and len(visited_triples) < M:
            cur = queue.popleft()
            for tidx in entity_to_triples[cur]:
                if tidx in visited_triples:
                    continue
                visited_triples.add(tidx)
                h, r, t = triples[tidx]
                for ent in (h, t):
                    if ent not in visited_entities:
                        visited_entities.add(ent)
                        queue.append(ent)
                if len(visited_triples) >= M:
                    break

        return [
            f"<subj> {h} <obj> {t} <rel> {r}"
            for i, (h, r, t) in enumerate(triples)
            if i in visited_triples
        ]

    # ------------------------------------------------------------------
    # Hop-based sampling (k-hop neighborhood)
    # ------------------------------------------------------------------

    def _hop_sample_subgraph(
        self,
        triple_groups: List[List[str]],
        hop: int,
        start_entity: str
    ) -> List[str]:

        triples = []
        for group in triple_groups:
            for t in group:
                triples.append(self._parse_triple(t))

        entity_to_triples = defaultdict(list)
        for idx, (h, r, t) in enumerate(triples):
            entity_to_triples[h].append(idx)
            entity_to_triples[t].append(idx)

        if start_entity not in entity_to_triples:
            return []

        visited_triples = set()
        visited_entities = {start_entity}
        queue = deque([(start_entity, 0)])

        while queue:
            entity, depth = queue.popleft()
            if depth >= hop:
                continue

            for tidx in entity_to_triples[entity]:
                if tidx in visited_triples:
                    continue
                visited_triples.add(tidx)
                h, r, t = triples[tidx]
                for nxt in (h, t):
                    if nxt not in visited_entities:
                        visited_entities.add(nxt)
                        queue.append((nxt, depth + 1))

        return [
            f"<subj> {h} <obj> {t} <rel> {r}"
            for i, (h, r, t) in enumerate(triples)
            if i in visited_triples
        ]

    # ------------------------------------------------------------------
    # Random walk sampling
    # ------------------------------------------------------------------

    def _random_walk_sampling(
        self,
        triple_groups: List[List[str]],
        start_entity: str,
        num_walks: int = 5,
        walk_length: int = 3
    ) -> List[str]:

        triples = [t for group in triple_groups for t in group]
        if not triples:
            return []

        neighbors = defaultdict(list)
        for t in triples:
            h, r, tail = self._parse_triple(t)
            neighbors[h].append((tail, t))
            neighbors[tail].append((h, t))

        if start_entity not in neighbors:
            return []

        sampled = set()
        for _ in range(num_walks):
            current = start_entity
            for _ in range(walk_length):
                if not neighbors[current]:
                    break
                nxt, triple = self.rng.choice(neighbors[current])
                sampled.add(triple)
                current = nxt

        return list(sampled)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        # 自动选择输入列
        if hasattr(self, "input_key") and self.input_key in dataframe.columns:
            chosen_input_key = self.input_key
        elif "triple" in dataframe.columns:
            chosen_input_key = "triple"
        elif "tuple" in dataframe.columns:
            chosen_input_key = "tuple"
        else:
            raise ValueError(
                "Missing required input column: neither 'triple' nor 'tuple' found in dataframe"
            )
        self.input_key = chosen_input_key

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key: str = "subgraph",
        sampling_type: str = "hop",     # bfs | rw | hop
        start_entity: Optional[str] = None,
        M: int = 5,
        hop: int = 2,
        num_walks: int = 5,
        walk_length: int = 3
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        all_tuples = []

        if len(dataframe) == 0:
            raise ValueError("DataFrame is empty.")

        elif len(dataframe) == 1:
            # 只有一行
            row_data = dataframe[self.input_key].iloc[0]
            if isinstance(row_data, list):
                all_tuples = row_data
            else:
                raise ValueError("Row data must be List[str]")

        else:
            # 多行 → 合并
            for row in dataframe[self.input_key]:
                if isinstance(row, list):
                    all_tuples.extend(row)

        triple_groups = [all_tuples]           

        # 1. determine start entity
        all_entities = self._collect_entities(triple_groups)
        if not all_entities:
            raise ValueError("No entities found in input triples")

        if start_entity is None:
            start_entity = all_entities

        # 2. sampling
        all_sampled_subgraphs = []
        for entity in start_entity:
            if sampling_type == "bfs":
                sampled = self._bfs_sample_subgraph(
                    triple_groups, M=M, start_entity=entity
                )

            elif sampling_type == "hop":
                sampled = self._hop_sample_subgraph(
                    triple_groups, hop=hop, start_entity=entity
                )

            elif sampling_type == "rw":
                sampled = self._random_walk_sampling(
                    triple_groups,
                    start_entity=entity,
                    num_walks=num_walks,
                    walk_length=walk_length
                )

            else:
                raise ValueError(
                    f"Unknown sampling_type: {sampling_type}, "
                    f"expected one of ['bfs', 'hop', 'rw']"
                )

            all_sampled_subgraphs.append({
                self.output_key: sampled
            })

        # 3. write back
        df = pd.DataFrame()
        df[self.output_key] = [o[self.output_key] for o in all_sampled_subgraphs]
        output_file = storage.write(df)
        self.logger.info(
            f"KG sampling done | type={sampling_type} | saved to {output_file}"
        )

        return [self.output_key]