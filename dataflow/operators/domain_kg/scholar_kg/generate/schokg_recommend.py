from dataflow.prompts.diverse_kg.schokg import SchoKGRecommendPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import prompt_restrict
import random
from typing import Any, Dict, List, Optional
import json
import re
from collections import defaultdict


@prompt_restrict(
    SchoKGRecommendPrompt,
)
@OPERATOR_REGISTRY.register()
class SchoKGRecommendOperator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        max_hop: int = 3,
        max_candidate_nodes: int = 20,
        max_paths_per_node: int = 3,
        prompt_template=None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.max_hop = max_hop
        self.max_candidate_nodes = max_candidate_nodes
        self.max_paths_per_node = max_paths_per_node
        self.logger = get_logger()
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else SchoKGRecommendPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en"):
        if lang == "zh":
            return (
                "SchoKGRecommendOperator 用于根据用户问题推荐学者知识图谱中的目标节点。",
                "输入: query, triple, entity_class; 输出: recommended_node, recommendation_reason"
            )
        else:
            return (
                "SchoKGRecommendOperator recommends target nodes from a scholarly knowledge graph.",
                "Input: query, triple, entity_class; Output: recommended_node, recommendation_reason"
            )

    def _parse_triple(self, triple_str: str) -> Optional[Dict[str, str]]:
        subj_match = re.search(r"<subj>\s*(.+?)\s*(?=<obj>)", triple_str)
        obj_match = re.search(r"<obj>\s*(.+?)\s*(?=<rel>)", triple_str)
        rel_match = re.search(r"<rel>\s*(.+)$", triple_str)

        if not subj_match or not obj_match or not rel_match:
            return None

        return {
            "subj": subj_match.group(1).strip(),
            "obj": obj_match.group(1).strip(),
            "rel": rel_match.group(1).strip(),
        }

    def _build_graph(
        self,
        triples: List[str],
        entity_classes: List[List[str]],
    ):
        adj = defaultdict(list)
        entity_type_map = defaultdict(set)
        entities = set()

        for idx, triple in enumerate(triples):
            parsed = self._parse_triple(triple)
            if not parsed:
                continue

            classes = entity_classes[idx] if idx < len(entity_classes) else []
            subj_types = []
            obj_types = []

            if isinstance(classes, list):
                if len(classes) >= 1:
                    subj_types = [str(classes[0])]
                if len(classes) >= 2:
                    obj_types = [str(classes[1])]

            edge_id = f"{idx}:{triple}"

            forward_edge = {
                "edge_id": edge_id,
                "subj": parsed["subj"],
                "obj": parsed["obj"],
                "rel": parsed["rel"],
                "path_repr": f"<subj> {parsed['subj']} <obj> {parsed['obj']} <rel> {parsed['rel']}",
            }
            reverse_edge = {
                "edge_id": edge_id,
                "subj": parsed["obj"],
                "obj": parsed["subj"],
                "rel": parsed["rel"],
                "path_repr": f"<subj> {parsed['obj']} <obj> {parsed['subj']} <rel> {parsed['rel']}",
            }

            adj[parsed["subj"]].append((parsed["obj"], forward_edge))
            adj[parsed["obj"]].append((parsed["subj"], reverse_edge))

            for subj_type in subj_types:
                entity_type_map[parsed["subj"]].add(subj_type)
            for obj_type in obj_types:
                entity_type_map[parsed["obj"]].add(obj_type)

            entities.add(parsed["subj"])
            entities.add(parsed["obj"])

        return adj, entity_type_map, sorted(entities)

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _get_token_set(self, text: str) -> set:
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(normalized.split())

    def _match_query_entities(self, query: str, entities: List[str]) -> List[str]:
        query_normalized = self._normalize_text(query)
        query_tokens = self._get_token_set(query)
        matched = []

        for entity in entities:
            entity_normalized = self._normalize_text(entity)
            entity_tokens = self._get_token_set(entity)

            score = 0
            if entity_normalized and entity_normalized in query_normalized:
                score += 10

            if entity_tokens:
                overlap = len(entity_tokens & query_tokens)
                if overlap > 0:
                    score += overlap

            if score > 0:
                matched.append((entity, score, len(entity_normalized)))

        matched.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [item[0] for item in matched]

    def _path_to_string(self, path_edges: List[Dict[str, str]]) -> str:
        return " || ".join([edge["path_repr"] for edge in path_edges])

    def _collect_candidate_nodes(
        self,
        query: str,
        triples: List[str],
        entity_classes: List[List[str]],
        target_type: str,
    ) -> List[Dict[str, Any]]:
        adj, entity_type_map, entities = self._build_graph(triples, entity_classes)
        matched_entities = self._match_query_entities(query, entities)

        if not matched_entities:
            return []

        candidate_map = {}
        seed_entities = set(matched_entities[:4])

        def add_candidate(node: str, path_str: str):
            node_types = sorted(entity_type_map.get(node, set()))
            if target_type not in node_types:
                return

            candidate = candidate_map.setdefault(
                node,
                {
                    "node": node,
                    "type": target_type,
                    "supporting_paths": [],
                }
            )

            if path_str not in candidate["supporting_paths"]:
                candidate["supporting_paths"].append(path_str)

        def dfs(current_node, current_path, used_edges, seed_node):
            if len(current_path) >= self.max_hop:
                return

            for next_node, edge in adj[current_node]:
                if edge["edge_id"] in used_edges:
                    continue

                new_path = current_path + [edge]
                path_str = self._path_to_string(new_path)

                if next_node != seed_node and next_node not in seed_entities:
                    add_candidate(next_node, path_str)

                dfs(
                    next_node,
                    new_path,
                    used_edges | {edge["edge_id"]},
                    seed_node,
                )

        for seed in matched_entities[:2]:
            dfs(seed, [], set(), seed)

        candidate_nodes = list(candidate_map.values())

        for candidate in candidate_nodes:
            candidate["supporting_paths"] = candidate["supporting_paths"][: self.max_paths_per_node]

        query_tokens = self._get_token_set(query)

        def score(item: Dict[str, Any]):
            node_tokens = self._get_token_set(item["node"])
            overlap = len(node_tokens & query_tokens)
            path_count = len(item["supporting_paths"])
            min_hop = min((path.count("||") + 1 for path in item["supporting_paths"]), default=self.max_hop + 1)
            return (-path_count, -overlap, min_hop, item["node"])

        candidate_nodes = sorted(candidate_nodes, key=score)
        return candidate_nodes[: self.max_candidate_nodes]

    def _recommend_with_prompt(
        self,
        query: str,
        target_type: str,
        candidate_nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        user_inputs = [
            self.prompt_template.build_prompt(query, target_type, candidate_nodes)
        ]
        system_prompt = self.prompt_template.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=user_inputs,
            system_prompt=system_prompt,
        )

        return self._parse_llm_response(responses[0])

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            data = json.loads(cleaned)
            recommended_node = data.get("recommended_node", [])
            recommendation_reason = data.get("recommendation_reason", "")

            if not isinstance(recommended_node, list):
                recommended_node = []
            if not isinstance(recommendation_reason, str):
                recommendation_reason = str(recommendation_reason)

            return {
                "recommended_node": recommended_node,
                "recommendation_reason": recommendation_reason,
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return {
                "recommended_node": [],
                "recommendation_reason": "",
            }

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key_query, self.input_key_triple, self.input_key_class]
        forbidden_keys = [
            self.output_key_node,
            self.output_key_reason,
        ]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_query: str = "query",
        input_key_triple: str = "triple",
        input_key_class: str = "entity_class",
        target_type: str = "Author",
        output_key_node: str = "recommended_node",
        output_key_reason: str = "recommendation_reason",
    ):
        self.input_key_query = input_key_query
        self.input_key_triple = input_key_triple
        self.input_key_class = input_key_class
        self.output_key_node = output_key_node
        self.output_key_reason = output_key_reason

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        queries = dataframe[self.input_key_query].tolist()
        triples_list = dataframe[self.input_key_triple].tolist()
        class_list = dataframe[self.input_key_class].tolist()

        recommended_nodes = []
        recommendation_reasons = []

        for query, triples, entity_classes in zip(queries, triples_list, class_list):
            if not isinstance(triples, list):
                triples = []
            if not isinstance(entity_classes, list):
                entity_classes = []

            candidate_nodes = self._collect_candidate_nodes(
                query=query,
                triples=triples,
                entity_classes=entity_classes,
                target_type=target_type,
            )
            result = self._recommend_with_prompt(query, target_type, candidate_nodes)

            recommended_nodes.append(result.get("recommended_node", []))
            recommendation_reasons.append(result.get("recommendation_reason", ""))

        dataframe[self.output_key_node] = recommended_nodes
        dataframe[self.output_key_reason] = recommendation_reasons

        output_file = storage.write(dataframe)
        self.logger.info(f"Scholarly node recommendation saved to {output_file}")

        return [output_key_node, output_key_reason]
