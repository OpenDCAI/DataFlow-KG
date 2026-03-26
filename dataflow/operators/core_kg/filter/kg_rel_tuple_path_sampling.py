import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

import random
from typing import List, Dict
import re
from collections import defaultdict


@OPERATOR_REGISTRY.register()
class KGRelationTuplePathGenerator(OperatorABC):
    """
    Generate k-hop undirected paths from relation triples.

    Input triple format:
        "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by"

    Paths are generated WITHOUT considering edge direction.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        seed: int = 0,
        lang: str = "en",
        k: int = 2,
        max_paths_per_group: int = 100,
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.k = k
        self.max_paths = max_paths_per_group
        self.logger = get_logger()

        self.triple_pattern = re.compile(
            r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)"
        )

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRelationTuplePathGenerator 用于从关系三元组中生成无向 k-hop 路径。",
                "解析三元组并构建无向图，通过 DFS 枚举长度为 k 的路径，并对路径去重。",
                "输入列 triple 为关系三元组字符串列表，输出列名由参数 k 和 output_key_meta 拼接而成（如 k=2 时为 2_hop_paths），每行为一条路径字符串。"
            )
        else:
            return (
                "KGRelationTuplePathGenerator generates undirected k-hop paths from relation triples.",
                "Parses triples into an undirected graph and enumerates paths of length k via DFS with deduplication.",
                "Takes triple (List[str]) as input column and outputs a column named '{k}_{output_key_meta}' (e.g. '2_hop_paths' when k=2), where each row is a path string."
            )

    # =========================
    # Parsing
    # =========================
    def _parse_triple(self, t: str):
        """
        Parse a triple or multi-attribute triple:
        - Subject: immediately after <subj>
        - Object: immediately after <obj>
        - Relation: everything after <rel> (including all extra attributes)
        """
        triple_str = t.strip()

        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        if not subj_match:
            return None
        subj = subj_match.group(1).strip()

        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        if not obj_match:
            return None
        obj = obj_match.group(1).strip()

        rel_match = re.search(r"<rel>\s*(.+)$", triple_str)
        if not rel_match:
            return None
        rel = rel_match.group(1).strip()  # 保留所有属性信息

        return {
            "subj": subj,
            "obj": obj,
            "rel": rel,
            "raw": t,
        }

    # =========================
    # Path canonicalization (DEDUP CORE)
    # =========================
    def _canonicalize_path(self, path: List[Dict]) -> tuple:
        """
        Convert a path into a direction-invariant canonical form.
        Used for deduplication of undirected paths.
        """
        edge_keys = []
        for e in path:
            # direction-invariant edge key
            u, v = sorted([e["subj"], e["obj"]])
            edge_keys.append(f"{u}::{e['rel']}::{v}")

        # sort edges to remove order effect
        return tuple(sorted(edge_keys))

    # =========================
    # Core logic: k-hop paths
    # =========================
    def _generate_k_hop_paths(self, groups: List[List[str]]) -> List[List[str]]:
        all_outputs = []

        for group in groups:
            adj = defaultdict(list)

            for t in group:
                parsed = self._parse_triple(t)
                if not parsed:
                    continue

                u, v = parsed["subj"], parsed["obj"]

                # undirected graph
                adj[u].append((v, parsed))
                adj[v].append((u, parsed))

            paths = []
            seen_path_keys = set()

            # -------- DFS enumeration --------
            def dfs(current_node, current_path, used_edges):
                if len(current_path) == self.k:
                    key = self._canonicalize_path(current_path)
                    if key not in seen_path_keys:
                        seen_path_keys.add(key)
                        paths.append(
                            " || ".join([e["raw"] for e in current_path])
                        )
                    return

                for next_node, edge in adj[current_node]:
                    edge_id = id(edge)
                    if edge_id in used_edges:
                        continue

                    dfs(
                        next_node,
                        current_path + [edge],
                        used_edges | {edge_id},
                    )

            for start_node in adj.keys():
                dfs(start_node, [], set())
                if len(paths) >= self.max_paths:
                    break

            all_outputs.append(paths[: self.max_paths])

        return all_outputs

    # =========================
    # DataFrame interface
    # =========================
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

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        output_key_meta: str = "hop_paths",
    ):
        self.input_key, self.output_key_meta = input_key, output_key_meta

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        self.logger.info(f"Generating {self.k}-hop undirected paths with deduplication")

        # =========================
        # 合并所有数据
        # =========================
        all_tuples = []

        if len(df) == 0:
            raise ValueError("DataFrame is empty.")

        elif len(df) == 1:
            # 只有一行
            row_data = df[self.input_key].iloc[0]
            if isinstance(row_data, list):
                all_tuples = row_data
            else:
                raise ValueError("Row data must be List[str]")

        else:
            # 多行 → 合并
            for row in df[self.input_key]:
                if isinstance(row, list):
                    all_tuples.extend(row)

        # 生成 paths
        all_paths = self._generate_k_hop_paths([all_tuples])[0]  # 全部合并成一组

        # =========================
        # 包装成期望 JSON 形式
        # =========================
        output_list = []
        for path_str in all_paths:
            output_list.append({
                f"{self.k}_hop_paths": path_str
            })

        # 写回 storage
        data = pd.DataFrame()
        data[f"{self.k}_hop_paths"] = [o[f"{self.k}_hop_paths"] for o in output_list]
        output_file = storage.write(data)

        self.logger.info(f"K-hop paths saved to {output_file}")
        return [output_key_meta]