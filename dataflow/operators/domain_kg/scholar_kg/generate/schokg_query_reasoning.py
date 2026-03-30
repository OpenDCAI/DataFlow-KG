from dataflow.prompts.diverse_kg.schokg import SchoKGQueryReasoningPrompt
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
from itertools import combinations


@prompt_restrict(
    SchoKGQueryReasoningPrompt,
)
@OPERATOR_REGISTRY.register()
class SchoKGQueryReasoningOperator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        max_hop: int = 3,
        max_candidate_paths: int = 20,
        prompt_template=None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.max_hop = max_hop
        self.max_candidate_paths = max_candidate_paths
        self.logger = get_logger()
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else SchoKGQueryReasoningPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en"):
        if lang == "zh":
            return (
                "SchoKGQueryReasoningOperator 用于根据用户问题和学者知识图谱推理路径生成回答。",
                "输入: query, triple; 输出: reasoning_path, reasoning_answer"
            )
        else:
            return (
                "SchoKGQueryReasoningOperator answers user queries with scholarly KG reasoning paths.",
                "Input: query, triple; Output: reasoning_path, reasoning_answer"
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

    def _build_graph(self, triples: List[str]):
        adj = defaultdict(list)
        entities = set()

        for idx, triple in enumerate(triples):
            parsed = self._parse_triple(triple)
            if not parsed:
                continue

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

            entities.add(parsed["subj"])
            entities.add(parsed["obj"])

        return adj, sorted(entities)

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

    def _collect_paths_between(
        self,
        adj: Dict[str, List],
        start: str,
        end: str,
    ) -> List[str]:
        matched_paths = []
        seen_paths = set()

        def dfs(current_node, current_path, used_edges):
            if len(current_path) >= self.max_hop:
                return

            for next_node, edge in adj[current_node]:
                if edge["edge_id"] in used_edges:
                    continue

                new_path = current_path + [edge]

                if next_node == end:
                    path_str = self._path_to_string(new_path)
                    if path_str not in seen_paths:
                        seen_paths.add(path_str)
                        matched_paths.append(path_str)
                    continue

                dfs(
                    next_node,
                    new_path,
                    used_edges | {edge["edge_id"]},
                )

        dfs(start, [], set())
        return matched_paths

    def _collect_paths_from_seed(
        self,
        adj: Dict[str, List],
        seed: str,
    ) -> List[str]:
        matched_paths = []
        seen_paths = set()

        def dfs(current_node, current_path, used_edges):
            if len(current_path) >= self.max_hop:
                return

            for next_node, edge in adj[current_node]:
                if edge["edge_id"] in used_edges:
                    continue

                new_path = current_path + [edge]
                path_str = self._path_to_string(new_path)

                if path_str not in seen_paths:
                    seen_paths.add(path_str)
                    matched_paths.append(path_str)

                dfs(
                    next_node,
                    new_path,
                    used_edges | {edge["edge_id"]},
                )

        dfs(seed, [], set())
        return matched_paths

    def _rank_paths(self, query: str, paths: List[str]) -> List[str]:
        query_normalized = self._normalize_text(query)

        def score(path: str):
            path_normalized = self._normalize_text(path)
            overlap = sum(
                1 for token in query_normalized.split()
                if token and token in path_normalized
            )
            hop = path.count("||") + 1
            return (-overlap, hop, len(path))

        return sorted(paths, key=score)

    def _find_candidate_paths(self, query: str, triples: List[str]) -> List[str]:
        adj, entities = self._build_graph(triples)
        matched_entities = self._match_query_entities(query, entities)
        candidate_paths = []
        seen_paths = set()

        if not matched_entities:
            return []

        top_entities = matched_entities[:4]

        if len(top_entities) >= 2:
            for start, end in combinations(top_entities, 2):
                paths = self._collect_paths_between(adj, start, end)
                for path in paths:
                    if path not in seen_paths:
                        seen_paths.add(path)
                        candidate_paths.append(path)

        if not candidate_paths:
            for entity in top_entities[:2]:
                paths = self._collect_paths_from_seed(adj, entity)
                for path in paths:
                    if path not in seen_paths:
                        seen_paths.add(path)
                        candidate_paths.append(path)

        candidate_paths = self._rank_paths(query, candidate_paths)

        if len(candidate_paths) > self.max_candidate_paths:
            candidate_paths = candidate_paths[:self.max_candidate_paths]

        return candidate_paths

    def _answer_with_prompt(self, query: str, candidate_paths: List[str]) -> Dict[str, Any]:
        user_inputs = [
            self.prompt_template.build_prompt(query, candidate_paths)
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
            reasoning_path = data.get("reasoning_path", [])
            reasoning_answer = data.get("reasoning_answer", "")

            if not isinstance(reasoning_path, list):
                reasoning_path = []
            if not isinstance(reasoning_answer, str):
                reasoning_answer = str(reasoning_answer)

            return {
                "reasoning_path": reasoning_path,
                "reasoning_answer": reasoning_answer,
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return {
                "reasoning_path": [],
                "reasoning_answer": "",
            }

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key_query, self.input_key_triple]
        forbidden_keys = [self.output_key_path, self.output_key_answer]

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
        output_key_path: str = "reasoning_path",
        output_key_answer: str = "reasoning_answer",
    ):
        self.input_key_query = input_key_query
        self.input_key_triple = input_key_triple
        self.output_key_path = output_key_path
        self.output_key_answer = output_key_answer

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        queries = dataframe[self.input_key_query].tolist()
        triples_list = dataframe[self.input_key_triple].tolist()

        reasoning_paths = []
        reasoning_answers = []

        for query, triples in zip(queries, triples_list):
            if not isinstance(triples, list):
                triples = []

            candidate_paths = self._find_candidate_paths(query, triples)
            result = self._answer_with_prompt(query, candidate_paths)

            reasoning_paths.append(result.get("reasoning_path", []))
            reasoning_answers.append(result.get("reasoning_answer", ""))

        dataframe[self.output_key_path] = reasoning_paths
        dataframe[self.output_key_answer] = reasoning_answers

        output_file = storage.write(dataframe)
        self.logger.info(f"Scholarly query reasoning saved to {output_file}")

        return [output_key_path, output_key_answer]
