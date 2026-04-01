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
class MMKGRelationTuplePathGenerator(OperatorABC):

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

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MMKGRelationTuplePathGenerator 用于从多模态三元组中采样 k-hop 路径并补充视觉信息。",
                "输入: triple/tuple + vis_triple + img_dict; 输出: hop_paths + vis_triple + vis_url",
            )
        return (
            "MMKGRelationTuplePathGenerator is used to sample k-hop paths from multimodal triples and attach visual information.",
            "Input: triple/tuple + vis_triple + img_dict; Output: hop_paths + vis_triple + vis_url",
        )

    # =========================
    # Triple parser
    # =========================

    def _parse_triple(self, t: str):

        triple_str = t.strip()

        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        rel_match = re.search(r"<rel>\s*(.+)$", triple_str)

        if not subj_match or not obj_match or not rel_match:
            return None

        return {
            "subj": subj_match.group(1).strip(),
            "obj": obj_match.group(1).strip(),
            "rel": rel_match.group(1).strip(),
            "raw": t,
        }

    # =========================
    # Canonical path key
    # =========================

    def _canonicalize_path(self, path: List[Dict]):

        edge_keys = []

        for e in path:
            u, v = sorted([e["subj"], e["obj"]])
            edge_keys.append(f"{u}::{e['rel']}::{v}")

        return tuple(sorted(edge_keys))

    # =========================
    # Extract entities from path
    # =========================

    def _extract_entities_from_path(self, path_edges):

        entities = set()

        for e in path_edges:
            entities.add(e["subj"])
            entities.add(e["obj"])

        return entities

    # =========================
    # Extract vis info
    # =========================

    def _extract_vis_info(self, entities, vis_triples, img_dict):

        related_vis_triples = []
        vis_urls = []

        for vt in vis_triples:

            subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<rel>)", vt)
            obj_match = re.search(r"<obj>\s*(.+?)\s*$", vt)

            if not subj_match or not obj_match:
                continue

            entity = subj_match.group(1).strip()
            img_id = obj_match.group(1).strip()

            if entity in entities and img_id in img_dict:

                related_vis_triples.append(vt)
                vis_urls.append(img_dict[img_id])

        return related_vis_triples, vis_urls

    # =========================
    # Generate paths
    # =========================

    def _generate_k_hop_paths(self, groups: List[List[str]]):

        all_outputs = []

        for group in groups:

            adj = defaultdict(list)

            for t in group:

                parsed = self._parse_triple(t)

                if not parsed:
                    continue

                u, v = parsed["subj"], parsed["obj"]

                adj[u].append((v, parsed))
                adj[v].append((u, parsed))

            paths = []
            path_edges_list = []
            seen_path_keys = set()

            def dfs(current_node, current_path, used_edges):

                if len(current_path) == self.k:

                    key = self._canonicalize_path(current_path)

                    if key not in seen_path_keys:

                        seen_path_keys.add(key)

                        paths.append(
                            " || ".join([e["raw"] for e in current_path])
                        )

                        path_edges_list.append(current_path)

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

            all_outputs.append(
                list(zip(paths[:self.max_paths], path_edges_list[:self.max_paths]))
            )

        return all_outputs

    # =========================
    # DataFrame validation
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
                "Missing required input column: neither 'triple' nor 'tuple' found"
            )

        self.input_key = chosen_input_key

    # =========================
    # Run
    # =========================

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        output_key_meta: str = "hop_paths",
    ):

        self.input_key = input_key
        self.output_key_meta = output_key_meta

        df = storage.read("dataframe")

        self._validate_dataframe(df)

        self.logger.info(f"Generating {self.k}-hop paths with visual grounding")

        vis_triple_all = df.get("vis_triple", [[]])
        img_dict_all = df.get("img_dict", [{}])

        all_tuples = []

        if len(df) == 0:
            raise ValueError("DataFrame is empty.")

        elif len(df) == 1:

            row_data = df[self.input_key].iloc[0]

            if isinstance(row_data, list):
                all_tuples = row_data
            else:
                raise ValueError("Row data must be List[str]")

        else:

            for row in df[self.input_key]:
                if isinstance(row, list):
                    all_tuples.extend(row)

        all_paths = self._generate_k_hop_paths([all_tuples])[0]

        vis_triples = vis_triple_all[0]
        img_dict = img_dict_all[0]

        output_rows = []

        for path_str, path_edges in all_paths:

            entities = self._extract_entities_from_path(path_edges)

            related_vis_triples, vis_urls = self._extract_vis_info(
                entities,
                vis_triples,
                img_dict
            )

            output_rows.append({
                f"{self.k}_hop_paths": path_str,
                "vis_triple": related_vis_triples,
                "vis_url": vis_urls
            })

        data = pd.DataFrame(output_rows)

        output_file = storage.write(data)

        self.logger.info(f"K-hop paths with visual info saved to {output_file}")

        return [output_key_meta, "vis_triple", "vis_url"]
