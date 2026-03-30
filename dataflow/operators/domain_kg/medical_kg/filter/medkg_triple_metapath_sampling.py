import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

import random
from typing import List, Dict, Optional, Union
import re
from collections import defaultdict


@OPERATOR_REGISTRY.register()
class MedKGMetaPathGenerator(OperatorABC):
    """
    Match path instances in a medical graph using a user-defined meta-path rule.

    Input triple format:
        "<subj> lung cancer <obj> gefitinib <rel> treats"

    Input entity_class format:
        ["Disease", "Compound"]

    Example meta-path rule:
        "Disease -> treats -> Compound -> affects -> Gene"
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        seed: int = 0,
        lang: str = "en",
        max_paths_per_group: int = 100,
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.max_paths = max_paths_per_group
        self.logger = get_logger()

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGMetaPathGenerator 用于根据给定元路径规则匹配医学图中的实例路径。",
                "处理流程：解析三元组与实体类型 -> 构建无向图 -> 按元路径规则 DFS 匹配 -> 返回路径实例",
                "输入：triple, entity_class, meta_path_rule\n输出：matched_meta_path"
            )
        else:
            return (
                "MedKGMetaPathGenerator matches path instances using a user-defined meta-path rule.",
                "Steps: parse triples and entity types -> build undirected graph -> DFS match by meta-path rule -> return path instances",
                "Input: triple, entity_class, meta_path_rule\nOutput: matched_meta_path"
            )

    # =========================
    # Parsing
    # =========================
    def _parse_triple(self, triple: str, entity_class: Optional[List[str]] = None):
        triple_str = triple.strip()

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
        rel = rel_match.group(1).strip()

        subj_type, obj_type = self._normalize_entity_class(entity_class)

        return {
            "subj": subj,
            "obj": obj,
            "rel": rel,
            "subj_type": subj_type,
            "obj_type": obj_type,
            "raw": triple,
        }

    def _normalize_entity_class(self, entity_class: Optional[List[str]]) -> tuple[str, str]:
        if not isinstance(entity_class, list) or len(entity_class) == 0:
            return "Unknown", "Unknown"
        if len(entity_class) == 1:
            return str(entity_class[0]), str(entity_class[0])
        return str(entity_class[0]), str(entity_class[1])

    def _parse_meta_path_rule(self, meta_path_rule: Union[str, List[str]]) -> Dict[str, List[str]]:
        if isinstance(meta_path_rule, str):
            tokens = [token.strip() for token in meta_path_rule.split("->") if token.strip()]
        elif isinstance(meta_path_rule, list):
            tokens = [str(token).strip() for token in meta_path_rule if str(token).strip()]
        else:
            raise ValueError("meta_path_rule must be a string or a list")

        if len(tokens) < 3 or len(tokens) % 2 == 0:
            raise ValueError(
                "meta_path_rule must follow the format: "
                "'Type -> relation -> Type [-> relation -> Type ...]'"
            )

        entity_types = tokens[0::2]
        relation_types = tokens[1::2]

        return {
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    # =========================
    # Core logic
    # =========================
    def _build_graph(
        self,
        triples: List[str],
        entity_classes: List[List[str]],
    ) -> Dict[str, List[tuple[str, Dict]]]:
        adj = defaultdict(list)

        for idx, triple in enumerate(triples):
            entity_class = entity_classes[idx] if idx < len(entity_classes) else []
            parsed = self._parse_triple(triple, entity_class)
            if not parsed:
                continue

            edge_id = f"{idx}:{parsed['raw']}"
            u, v = parsed["subj"], parsed["obj"]

            forward_edge = {
                "edge_id": edge_id,
                "subj": parsed["subj"],
                "obj": parsed["obj"],
                "rel": parsed["rel"],
                "subj_type": parsed["subj_type"],
                "obj_type": parsed["obj_type"],
                "raw": parsed["raw"],
            }
            reverse_edge = {
                "edge_id": edge_id,
                "subj": parsed["obj"],
                "obj": parsed["subj"],
                "rel": parsed["rel"],
                "subj_type": parsed["obj_type"],
                "obj_type": parsed["subj_type"],
                "raw": parsed["raw"],
            }

            adj[u].append((v, forward_edge))
            adj[v].append((u, reverse_edge))

        return adj

    def _match_meta_path(
        self,
        triples: List[str],
        entity_classes: List[List[str]],
        meta_path_rule: Union[str, List[str]],
    ) -> List[str]:
        meta_path = self._parse_meta_path_rule(meta_path_rule)
        entity_types = meta_path["entity_types"]
        relation_types = meta_path["relation_types"]
        hop_count = len(relation_types)
        adj = self._build_graph(triples, entity_classes)

        matched_paths = []
        seen_paths = set()

        def dfs(current_node, depth, current_path, used_edges):
            if depth == hop_count:
                path_key = tuple(edge["edge_id"] for edge in current_path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    matched_paths.append(" || ".join([edge["raw"] for edge in current_path]))
                return

            expected_subj_type = entity_types[depth]
            expected_rel = relation_types[depth]
            expected_obj_type = entity_types[depth + 1]

            for next_node, edge in adj[current_node]:
                if edge["edge_id"] in used_edges:
                    continue
                if edge["subj_type"] != expected_subj_type:
                    continue
                if edge["rel"] != expected_rel:
                    continue
                if edge["obj_type"] != expected_obj_type:
                    continue

                dfs(
                    next_node,
                    depth + 1,
                    current_path + [edge],
                    used_edges | {edge["edge_id"]},
                )

        for start_node in adj.keys():
            dfs(start_node, 0, [], set())

        if len(matched_paths) > self.max_paths:
            self.rng.shuffle(matched_paths)
            matched_paths = matched_paths[: self.max_paths]

        return matched_paths

    # =========================
    # DataFrame interface
    # =========================
    def _validate_dataframe(self, dataframe: pd.DataFrame):
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

        if self.input_key_class not in dataframe.columns:
            raise ValueError(f"Missing required input column: {self.input_key_class}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        input_key_class: str = "entity_class",
        meta_path_rule: Union[str, List[str]] = None,
        output_key_meta: str = "matched_meta_path",
    ):
        self.input_key = input_key
        self.input_key_class = input_key_class
        self.output_key_meta = output_key_meta

        if meta_path_rule is None:
            raise ValueError("meta_path_rule must not be empty")

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        self.logger.info("Matching path instances by meta-path rule")

        all_triples = []
        all_entity_classes = []

        if len(df) == 0:
            raise ValueError("DataFrame is empty.")

        elif len(df) == 1:
            triple_data = df[self.input_key].iloc[0]
            class_data = df[self.input_key_class].iloc[0]
            if not isinstance(triple_data, list):
                raise ValueError("Row data must be List[str]")
            if not isinstance(class_data, list):
                raise ValueError("Entity class data must be List[List[str]]")
            all_triples = triple_data
            all_entity_classes = class_data

        else:
            for triples, entity_classes in zip(df[self.input_key], df[self.input_key_class]):
                if isinstance(triples, list):
                    all_triples.extend(triples)
                if isinstance(entity_classes, list):
                    all_entity_classes.extend(entity_classes)

        matched_paths = self._match_meta_path(
            triples=all_triples,
            entity_classes=all_entity_classes,
            meta_path_rule=meta_path_rule,
        )

        data = pd.DataFrame()
        data[self.output_key_meta] = matched_paths
        output_file = storage.write(data)

        self.logger.info(f"Matched meta-path instances saved to {output_file}")
        return [output_key_meta]
