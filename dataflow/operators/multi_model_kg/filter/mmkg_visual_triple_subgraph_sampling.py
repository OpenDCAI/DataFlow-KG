import pandas as pd
import random
import re
from typing import Dict, List, Optional
from collections import defaultdict, deque

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class MMKGEntityBasedSubgraphSampling(OperatorABC):

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

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MMKGEntityBasedSubgraphSampling 用于围绕实体采样多模态子图并补充视觉信息。",
                "输入: triple + vis_triple + img_dict; 输出: subgraph + vis_triple + vis_url",
            )
        return (
            "MMKGEntityBasedSubgraphSampling is used to sample multimodal subgraphs around entities and attach visual information.",
            "Input: triple + vis_triple + img_dict; Output: subgraph + vis_triple + vis_url",
        )

    # ------------------------------------------------
    # triple parser
    # ------------------------------------------------

    def _parse_triple(self, triple_str: str):

        triple_str = triple_str.strip()

        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        rel_match = re.search(r"<rel>\s*(.+)$", triple_str)

        if not subj_match or not obj_match or not rel_match:
            raise ValueError(f"Invalid triple: {triple_str}")

        subj = subj_match.group(1).strip()
        obj = obj_match.group(1).strip()
        rel = rel_match.group(1).strip()

        return subj, rel, obj

    # ------------------------------------------------
    # entity collection
    # ------------------------------------------------

    def _collect_entities(self, triple_groups):

        entities = set()

        for group in triple_groups:
            for t in group:
                h, r, tail = self._parse_triple(t)
                entities.add(h)
                entities.add(tail)

        return list(entities)

    # ------------------------------------------------
    # BFS sampling
    # ------------------------------------------------

    def _bfs_sample_subgraph(self, triple_groups, M, start_entity):

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

    # ------------------------------------------------
    # Hop sampling
    # ------------------------------------------------

    def _hop_sample_subgraph(self, triple_groups, hop, start_entity):

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

    # ------------------------------------------------
    # Extract image URLs
    # ------------------------------------------------

    def _extract_vis_urls(self, entities, vis_triples, img_dict):

        vis_urls = []
        seen = set()

        for vt in vis_triples:

            subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<rel>)", vt)
            obj_match = re.search(r"<obj>\s*(.+?)\s*$", vt)

            if not subj_match or not obj_match:
                continue

            entity = subj_match.group(1).strip()
            img_id = obj_match.group(1).strip()

            if entity in entities and img_id in img_dict:

                url = img_dict[img_id]

                if url not in seen:
                    vis_urls.append(url)
                    seen.add(url)

        return vis_urls

    # ------------------------------------------------
    # Run
    # ------------------------------------------------

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "subgraph",
        vis_triple_key: str = "vis_triple",
        sampling_type: str = "hop",
        start_entity: Optional[str] = None,
        M: int = 5,
        hop: int = 2
    ):

        dataframe = storage.read("dataframe")

        vis_triple_all = dataframe.get(vis_triple_key, [[] for _ in range(len(dataframe))])
        img_dict_all = dataframe.get("img_dict", [{} for _ in range(len(dataframe))])

        row_data = dataframe[input_key].iloc[0]

        triple_groups = [row_data]

        all_entities = self._collect_entities(triple_groups)

        if start_entity is None:
            start_entity = self.rng.choice(all_entities)

        all_sampled_subgraphs = []

        for idx, entity in enumerate(all_entities):

            if sampling_type == "bfs":

                sampled = self._bfs_sample_subgraph(
                    triple_groups, M=M, start_entity=entity
                )

            else:

                sampled = self._hop_sample_subgraph(
                    triple_groups, hop=hop, start_entity=entity
                )

            # -----------------------------------
            # collect entities in subgraph
            # -----------------------------------

            entities_in_subgraph = set()

            for t in sampled:

                subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", t)
                obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", t)

                if subj_match:
                    entities_in_subgraph.add(subj_match.group(1).strip())

                if obj_match:
                    entities_in_subgraph.add(obj_match.group(1).strip())

            # -----------------------------------
            # filter vis triples
            # -----------------------------------

            related_vis_triples = []

            vis_triples = vis_triple_all[0] if len(vis_triple_all) > 0 else []

            for vt in vis_triples:

                subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<rel>)", vt)

                if subj_match:

                    if subj_match.group(1).strip() in entities_in_subgraph:
                        related_vis_triples.append(vt)

            # -----------------------------------
            # extract image urls
            # -----------------------------------

            img_dict = img_dict_all[0] if len(img_dict_all) > 0 else {}

            vis_urls = self._extract_vis_urls(
                entities_in_subgraph,
                related_vis_triples,
                img_dict
            )

            all_sampled_subgraphs.append({

                output_key: sampled,
                vis_triple_key: related_vis_triples,
                "vis_url": vis_urls

            })

        df = pd.DataFrame(all_sampled_subgraphs)

        output_file = storage.write(df)

        self.logger.info(
            f"KG sampling with vis_triple & vis_url done | type={sampling_type} | saved to {output_file}"
        )

        return [output_key, vis_triple_key, "vis_url"]
