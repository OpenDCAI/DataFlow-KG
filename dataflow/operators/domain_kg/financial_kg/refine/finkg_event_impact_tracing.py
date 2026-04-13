import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.finkg import (
    FinKGEventImpactTracingPrompt,
    FinKGEventQueryExtractionPrompt,
)
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

class FinKGEventImpactTracingLLM:

    _REL_PATTERN = re.compile(
        r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*?)\s*<time>\s*(.*)"
    )

    EVENT_RELATIONS = {
        "defaults_on",
        "fined_by",
        "sanctioned_by",
        "downgraded_by",
        "upgraded_by",
        "announced_by",
        "triggers",
        "results_in",
        "affects",
        "participates_in",
    }

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC],
        lang: str = "en",
        k_hops: int = 2,
        max_context_tuples: int = 24,
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.k_hops = max(1, int(k_hops))
        self.max_context_tuples = max(1, int(max_context_tuples))
        self.logger = get_logger()
        self.prompt_template = FinKGEventImpactTracingPrompt(lang=lang)
        self.event_query_prompt = FinKGEventQueryExtractionPrompt(lang=lang)

    @classmethod
    def _parse_tuple(cls, tuple_str: str) -> Optional[Dict[str, str]]:
        if not isinstance(tuple_str, str):
            return None

        matched = cls._REL_PATTERN.search(tuple_str)
        if not matched:
            return None

        subj = matched.group(1).strip()
        obj = matched.group(2).strip()
        rel = matched.group(3).strip()
        time_value = matched.group(4).strip() or "NA"

        if not subj or not obj or not rel:
            return None

        return {
            "subj": subj,
            "obj": obj,
            "rel": rel,
            "time": time_value,
            "raw": tuple_str,
        }

    def _find_related_tuples(
        self,
        tuples: List[str],
        entity1: str,
        entity2: str,
    ) -> List[str]:
        if not isinstance(tuples, list) or not tuples:
            return []

        edges: List[Dict[str, str]] = []
        for tuple_str in tuples:
            parsed = self._parse_tuple(tuple_str)
            if parsed:
                edges.append(parsed)

        if not edges:
            return []

        entity_to_edge_ids: Dict[str, List[int]] = {}
        for index, edge in enumerate(edges):
            entity_to_edge_ids.setdefault(edge["subj"], []).append(index)
            entity_to_edge_ids.setdefault(edge["obj"], []).append(index)

        frontier = {entity1, entity2}
        visited = set(frontier)
        selected = set()

        for _ in range(self.k_hops):
            next_frontier = set()
            for entity in frontier:
                for edge_id in entity_to_edge_ids.get(entity, []):
                    selected.add(edge_id)
                    edge = edges[edge_id]
                    for neighbor in (edge["subj"], edge["obj"]):
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier

        return [edges[index]["raw"] for index in sorted(selected)]

    def llm_query(
        self,
        raw_data: List[Dict[str, Any]],
        ontology: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.llm_serving is None:
            raise ValueError("llm_serving is required for event impact tracing")

        results = []
        for data in tqdm(raw_data, desc="Trace event impact"):
            raw_event_text = self._normalize_text(data.get("raw_event_text"))
            target_event = self._normalize_text(data.get("target_event"))
            target_entity = self._normalize_text(data.get("target_entity"))
            tuples = self._normalize_tuples(data.get("tuple"))
            detected_event = ""
            anchor_entities = []

            if raw_event_text:
                extracted = self._extract_event_query(
                    raw_event_text=raw_event_text,
                    ontology=ontology,
                )
                detected_event = extracted.get("target_event", "")
                anchor_entities.extend(extracted.get("anchor_entities", []))

            if target_entity:
                anchor_entities.append(target_entity)

            anchor_entities = self._ground_entities_to_tuples(
                entities=anchor_entities,
                tuples=tuples,
                raw_event_text=raw_event_text,
            )
            if not target_event:
                target_event = detected_event or raw_event_text

            relevant_tuples = self._select_relevant_tuples(
                tuples=tuples,
                target_event=target_event,
                anchor_entities=anchor_entities,
            )

            if not relevant_tuples:
                result = self._empty_result()
                result["detected_event"] = detected_event or target_event
                results.append(result)
                continue

            user_prompt = self.prompt_template.build_prompt(
                target_event=target_event,
                target_entity=", ".join(anchor_entities),
                tuple_text="\n".join(relevant_tuples),
                raw_event_text=raw_event_text,
            )
            system_prompt = self.prompt_template.build_system_prompt(ontology)

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )
            parsed = self._parse_response(responses[0] if responses else "")
            parsed["impacted_entities"] = self._ground_entities_to_tuples(
                entities=parsed.get("impacted_entities", []),
                tuples=relevant_tuples,
            )
            parsed["event_impact_paths"] = self._sanitize_paths(
                paths=parsed.get("event_impact_paths", []),
                evidence_tuples=relevant_tuples,
            )
            if not parsed["event_impact_paths"]:
                parsed["event_impact_paths"] = self._build_fallback_paths(
                    evidence_tuples=relevant_tuples,
                    target_event=target_event,
                    focus_entities=anchor_entities + parsed["impacted_entities"],
                )
            parsed["detected_event"] = detected_event or target_event
            results.append(parsed)

        return results

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    def _normalize_tuples(self, tuples: Any) -> List[str]:
        if not isinstance(tuples, list):
            return []
        return [item.strip() for item in tuples if isinstance(item, str) and item.strip()]

    def _extract_event_query(
        self,
        raw_event_text: str,
        ontology: Dict[str, Any],
    ) -> Dict[str, Any]:
        empty = {"target_event": "", "anchor_entities": []}
        if not raw_event_text:
            return empty

        user_prompt = self.event_query_prompt.build_prompt(raw_event_text=raw_event_text)
        system_prompt = self.event_query_prompt.build_system_prompt(ontology)

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        try:
            data = json.loads(self._strip_code_fence(responses[0] if responses else ""))
        except Exception as exc:
            self.logger.warning(f"Failed to parse event query extraction response: {exc}")
            return empty

        target_event = self._safe_string(data.get("target_event"))
        anchor_entities = self._safe_string_list(data.get("anchor_entities"))
        return {
            "target_event": target_event,
            "anchor_entities": anchor_entities,
        }

    def _ground_entities_to_tuples(
        self,
        entities: List[str],
        tuples: List[str],
        raw_event_text: str = "",
    ) -> List[str]:
        candidate_entities = self._collect_tuple_entities(tuples)
        grounded = []

        for entity in entities:
            if self._looks_like_generic_label(entity):
                continue
            match = self._best_entity_match(entity, candidate_entities)
            if match:
                grounded.append(match)

        if raw_event_text:
            grounded.extend(
                self._entities_explicitly_mentioned_in_text(
                    raw_event_text=raw_event_text,
                    candidate_entities=candidate_entities,
                )
            )

        return self._dedupe_keep_order(grounded)

    def _select_relevant_tuples(
        self,
        tuples: List[str],
        target_event: str,
        anchor_entities: List[str],
    ) -> List[str]:
        selected = []

        for anchor_entity in anchor_entities:
            selected.extend(
                self._find_related_tuples(
                    tuples=tuples,
                    entity1=anchor_entity,
                    entity2=anchor_entity,
                )
            )

        selected.extend(self._match_query_tuples(tuples, target_event))
        selected.extend(self._event_relation_tuples(tuples))

        if not selected:
            selected = tuples[: self.max_context_tuples]

        return self._dedupe_keep_order(selected)[: self.max_context_tuples]

    def _collect_tuple_entities(self, tuples: List[str]) -> List[str]:
        entities = []
        for tuple_str in tuples:
            parsed = self._parse_tuple(tuple_str)
            if not parsed:
                continue
            entities.extend([parsed["subj"], parsed["obj"]])
        return self._dedupe_keep_order(
            [entity for entity in entities if isinstance(entity, str) and entity.strip()]
        )

    def _looks_like_generic_label(self, entity: str) -> bool:
        normalized = self._normalize_query(entity)
        if not normalized:
            return True
        if re.fullmatch(r"\d{4}", normalized):
            return True
        generic_terms = {
            "event",
            "defaultevent",
            "regulatoryaction",
            "occurrence",
            "institution",
            "corporatebond",
            "security",
            "company",
            "corporation",
            "organization",
        }
        compact = normalized.replace(" ", "")
        return compact in generic_terms

    def _best_entity_match(
        self,
        entity: str,
        candidate_entities: List[str],
    ) -> str:
        normalized_entity = self._normalize_query(entity)
        if not normalized_entity:
            return ""

        exact_matches = []
        fuzzy_matches = []
        entity_tokens = set(normalized_entity.split())

        for candidate in candidate_entities:
            normalized_candidate = self._normalize_query(candidate)
            if not normalized_candidate:
                continue
            if normalized_candidate == normalized_entity:
                exact_matches.append(candidate)
                continue
            if (
                normalized_entity in normalized_candidate
                or normalized_candidate in normalized_entity
            ):
                fuzzy_matches.append(candidate)
                continue
            candidate_tokens = set(normalized_candidate.split())
            if entity_tokens and len(entity_tokens & candidate_tokens) >= min(2, len(entity_tokens)):
                fuzzy_matches.append(candidate)

        if exact_matches:
            return min(exact_matches, key=len)
        if fuzzy_matches:
            return min(fuzzy_matches, key=lambda item: (abs(len(item) - len(entity)), len(item)))
        return ""

    def _entities_explicitly_mentioned_in_text(
        self,
        raw_event_text: str,
        candidate_entities: List[str],
    ) -> List[str]:
        normalized_text = self._normalize_query(raw_event_text)
        mentioned = []

        for candidate in candidate_entities:
            normalized_candidate = self._normalize_query(candidate)
            if not normalized_candidate:
                continue
            if normalized_candidate in normalized_text:
                mentioned.append(candidate)

        return mentioned

    def _match_query_tuples(self, tuples: List[str], query: str) -> List[str]:
        query = self._normalize_query(query)
        if not query:
            return []

        query_tokens = set(query.split())
        scored = []

        for tuple_str in tuples:
            norm_tuple = self._normalize_query(tuple_str)
            score = 0
            if query in norm_tuple:
                score += 10
            if query_tokens:
                score += len(query_tokens & set(norm_tuple.split()))
            if score > 0:
                scored.append((score, tuple_str))

        scored.sort(key=lambda item: (-item[0], len(item[1])))
        return [item[1] for item in scored[: self.max_context_tuples]]

    def _event_relation_tuples(self, tuples: List[str]) -> List[str]:
        matched = []
        for tuple_str in tuples:
            parsed = self._parse_tuple(tuple_str)
            if parsed and parsed["rel"] in self.EVENT_RELATIONS:
                matched.append(tuple_str)
        return matched

    def _normalize_query(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[_-]+", " ", text)
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def _strip_code_fence(self, response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        result = self._empty_result()

        try:
            data = json.loads(self._strip_code_fence(response))
        except Exception as exc:
            self.logger.warning(f"Failed to parse event impact response: {exc}")
            return result

        result["event_answer"] = self._safe_string(data.get("analysis_summary"))
        result["impacted_entities"] = self._safe_string_list(data.get("impacted_entities"))
        result["impact_types"] = self._safe_string_list(data.get("impact_types"))
        result["event_impact_paths"] = self._safe_string_list(data.get("impact_paths"))
        result["event_impact_confidence"] = self._safe_confidence(data.get("confidence"))
        return result

    def _safe_string(self, value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _safe_string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    def _safe_confidence(self, value: Any) -> str:
        if not isinstance(value, str):
            return "low"
        value = value.strip().lower()
        return value if value in {"high", "medium", "low"} else "low"

    def _sanitize_paths(
        self,
        paths: List[str],
        evidence_tuples: List[str],
    ) -> List[str]:
        if not isinstance(paths, list) or not evidence_tuples:
            return []

        evidence_set = {item.strip() for item in evidence_tuples if isinstance(item, str)}
        sanitized = []

        for path in paths:
            if not isinstance(path, str):
                continue
            segments = [segment.strip() for segment in path.split("||") if segment.strip()]
            if not segments:
                continue
            if all(segment in evidence_set for segment in segments):
                sanitized.append(" || ".join(segments))

        return self._dedupe_keep_order(sanitized)

    def _build_fallback_paths(
        self,
        evidence_tuples: List[str],
        target_event: str,
        focus_entities: List[str],
    ) -> List[str]:
        parsed_tuples = []
        focus_set = {
            entity for entity in focus_entities
            if isinstance(entity, str) and entity.strip()
        }
        normalized_event = self._normalize_query(target_event)

        for tuple_str in evidence_tuples:
            parsed = self._parse_tuple(tuple_str)
            if parsed:
                parsed_tuples.append((tuple_str, parsed))

        scored_paths = []
        for idx_a, (tuple_a, parsed_a) in enumerate(parsed_tuples):
            entities_a = {parsed_a["subj"], parsed_a["obj"]}
            for idx_b in range(idx_a + 1, len(parsed_tuples)):
                tuple_b, parsed_b = parsed_tuples[idx_b]
                entities_b = {parsed_b["subj"], parsed_b["obj"]}
                shared_entities = entities_a & entities_b
                if not shared_entities:
                    continue
                if (
                    parsed_a["rel"] not in self.EVENT_RELATIONS
                    and parsed_b["rel"] not in self.EVENT_RELATIONS
                ):
                    continue

                combined_entities = entities_a | entities_b
                score = len(shared_entities)
                if focus_set and combined_entities & focus_set:
                    score += 3
                if parsed_a["rel"] in self.EVENT_RELATIONS:
                    score += 1
                if parsed_b["rel"] in self.EVENT_RELATIONS:
                    score += 1
                if normalized_event:
                    norm_a = self._normalize_query(tuple_a)
                    norm_b = self._normalize_query(tuple_b)
                    if normalized_event in norm_a or normalized_event in norm_b:
                        score += 2

                scored_paths.append((score, f"{tuple_a} || {tuple_b}"))

        if not scored_paths:
            for tuple_str, parsed in parsed_tuples:
                tuple_entities = {parsed["subj"], parsed["obj"]}
                if parsed["rel"] not in self.EVENT_RELATIONS:
                    continue
                score = 1
                if focus_set and tuple_entities & focus_set:
                    score += 2
                if normalized_event and normalized_event in self._normalize_query(tuple_str):
                    score += 2
                scored_paths.append((score, tuple_str))

        scored_paths.sort(key=lambda item: (-item[0], len(item[1])))
        return self._dedupe_keep_order(
            [path for _, path in scored_paths]
        )[: min(6, self.max_context_tuples)]

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "event_answer": "",
            "impacted_entities": [],
            "impact_types": [],
            "event_impact_paths": [],
            "event_impact_confidence": "low",
            "detected_event": "",
        }


@prompt_restrict(
    FinKGEventQueryExtractionPrompt,
    FinKGEventImpactTracingPrompt,
)
@OPERATOR_REGISTRY.register()
class FinKGEventImpactTracing(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        k_hops: int = 2,
        max_context_tuples: int = 24,
    ):
        self.logger = get_logger()
        self.tracer = FinKGEventImpactTracingLLM(
            llm_serving=llm_serving,
            lang=lang,
            k_hops=k_hops,
            max_context_tuples=max_context_tuples,
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGEventImpactTracing 用于基于金融知识图谱追踪事件影响路径。",
                "输入: tuple + raw_event_text/target_event；输出: detected_event + event_answer。",
            )
        return (
            "FinKGEventImpactTracing is used to trace event impact paths with Financial KG.",
            "Input: tuple + raw_event_text/target_event. Output: detected_event + event_answer.",
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key_event, self.output_key]

        if len(set(forbidden_keys)) != len(forbidden_keys):
            raise ValueError("output_key_event and output_key must be different")

        missing = [key for key in required_keys if key not in dataframe.columns]
        has_event_text = "raw_event_text" in dataframe.columns
        has_event_key = "target_event" in dataframe.columns
        conflict = [key for key in forbidden_keys if key in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if not has_event_text and not has_event_key:
            raise ValueError(
                "Missing event input: provide either 'raw_event_text' or 'target_event'"
            )
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def process_batch(
        self,
        tuples_list: List[List[str]],
        ontology: Dict[str, Any],
        raw_event_texts: Optional[List[Any]] = None,
        target_events: Optional[List[Any]] = None,
        target_entities: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        num_rows = len(tuples_list)
        if raw_event_texts is None:
            raw_event_texts = [""] * num_rows
        if target_events is None:
            target_events = [""] * num_rows
        if target_entities is None:
            target_entities = [""] * num_rows

        raw_data = [
            {
                "tuple": tuples,
                "raw_event_text": raw_event_text,
                "target_event": target_event,
                "target_entity": target_entity,
            }
            for tuples, raw_event_text, target_event, target_entity in zip(
                tuples_list, raw_event_texts, target_events, target_entities
            )
        ]

        return self.tracer.llm_query(raw_data=raw_data, ontology=ontology)

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key_event: str = "detected_event",
        output_key: str = "event_answer",
    ) -> List[str]:
        self.input_key = input_key
        self.output_key_event = output_key_event
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        tuples_list = dataframe[self.input_key].tolist()

        raw_event_texts = None
        if "raw_event_text" in dataframe.columns:
            raw_event_texts = dataframe["raw_event_text"].tolist()

        target_events = None
        if "target_event" in dataframe.columns:
            target_events = dataframe["target_event"].tolist()

        target_entities = None
        if "target_entity" in dataframe.columns:
            target_entities = dataframe["target_entity"].tolist()

        ontology = load_finkg_ontology()

        outputs = self.process_batch(
            tuples_list=tuples_list,
            ontology=ontology,
            raw_event_texts=raw_event_texts,
            target_events=target_events,
            target_entities=target_entities,
        )

        dataframe[self.output_key_event] = [
            item.get("detected_event", "") for item in outputs
        ]
        dataframe[self.output_key] = [
            item.get("event_answer", "") for item in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Event impact tracing results saved to {output_file}")

        return [self.output_key_event, self.output_key]
