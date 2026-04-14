import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.finkg import FinKGEntityRiskAssessmentPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

class FinKGEntityRiskAssessmentLLM:

    _REL_PATTERN = re.compile(
        r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*?)\s*<time>\s*(.*)"
    )

    RISK_RELATIONS = {
        "guarantor_of",
        "guarantees",
        "defaults_on",
        "fined_by",
        "sanctioned_by",
        "downgraded_by",
        "violates",
        "borrows_from",
        "lends_to",
        "secured_by",
        "collateral_for",
        "counterparty_of",
        "affects",
        "controls",
        "major_shareholder_of",
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
        self.prompt_template = FinKGEntityRiskAssessmentPrompt(lang=lang)

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
            raise ValueError("llm_serving is required for risk exposure analysis")

        results = []
        for data in tqdm(raw_data, desc="Assess entity risk"):
            target_entity = self._normalize_text(data.get("target_entity"))
            tuples = self._normalize_tuples(data.get("tuple"))
            relevant_tuples = self._select_relevant_tuples(
                tuples=tuples,
                target_entity=target_entity,
            )

            if not relevant_tuples:
                results.append(self._empty_result())
                continue

            user_prompt = self.prompt_template.build_prompt(
                target_entity=target_entity,
                tuple_text="\n".join(relevant_tuples),
            )
            system_prompt = self.prompt_template.build_system_prompt(ontology)

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )
            parsed = self._parse_response(responses[0] if responses else "")
            parsed["risk_paths"] = self._sanitize_paths(
                paths=parsed.get("risk_paths", []),
                evidence_tuples=relevant_tuples,
            )
            if not parsed["risk_paths"]:
                parsed["risk_paths"] = self._build_fallback_paths(
                    evidence_tuples=relevant_tuples,
                    target_entity=target_entity,
                    risk_entities=parsed.get("risk_entities", []),
                )
            if parsed["risk_score"] <= 0:
                parsed["risk_score"] = self._estimate_risk_score(
                    risk_types=parsed.get("risk_types", []),
                    risk_entities=parsed.get("risk_entities", []),
                    risk_paths=parsed.get("risk_paths", []),
                )
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

    def _select_relevant_tuples(
        self,
        tuples: List[str],
        target_entity: str,
    ) -> List[str]:
        selected = []

        if target_entity:
            selected.extend(
                self._find_related_tuples(
                    tuples=tuples,
                    entity1=target_entity,
                    entity2=target_entity,
                )
            )

        selected.extend(self._risk_relation_tuples(selected or tuples))

        if not selected:
            selected = tuples[: self.max_context_tuples]

        return self._dedupe_keep_order(selected)[: self.max_context_tuples]

    def _risk_relation_tuples(self, tuples: List[str]) -> List[str]:
        matched = []
        for tuple_str in tuples:
            parsed = self._parse_tuple(tuple_str)
            if parsed and parsed["rel"] in self.RISK_RELATIONS:
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
            self.logger.warning(f"Failed to parse risk exposure response: {exc}")
            return result

        result["risk_answer"] = self._safe_string(data.get("analysis_summary"))
        result["risk_types"] = self._safe_string_list(data.get("risk_types"))
        result["risk_entities"] = self._safe_string_list(
            data.get("risk_entities") or data.get("exposure_entities")
        )
        result["risk_paths"] = self._safe_string_list(data.get("risk_paths"))
        result["risk_score"] = self._safe_score(data.get("risk_score"))
        if result["risk_score"] <= 0:
            result["risk_score"] = self._score_from_confidence(data.get("confidence"))
        return result

    def _safe_string(self, value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _safe_string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    def _safe_score(self, value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, (int, float)):
            return max(0, min(100, int(round(value))))
        if isinstance(value, str):
            matched = re.search(r"-?\d+(?:\.\d+)?", value)
            if matched:
                return max(0, min(100, int(round(float(matched.group(0))))))
        return 0

    def _score_from_confidence(self, value: Any) -> int:
        if not isinstance(value, str):
            return 0
        value = value.strip().lower()
        if value == "high":
            return 80
        if value == "medium":
            return 60
        if value == "low":
            return 35
        return 0

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
        target_entity: str,
        risk_entities: List[str],
    ) -> List[str]:
        parsed_tuples = []
        focus_entities = {
            entity.strip()
            for entity in risk_entities
            if isinstance(entity, str) and entity.strip()
        }
        if target_entity:
            focus_entities.add(target_entity)

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
                    parsed_a["rel"] not in self.RISK_RELATIONS
                    and parsed_b["rel"] not in self.RISK_RELATIONS
                ):
                    continue

                combined_entities = entities_a | entities_b
                score = len(shared_entities)
                if target_entity and target_entity in combined_entities:
                    score += 3
                if focus_entities and combined_entities & focus_entities:
                    score += 2
                if parsed_a["rel"] in self.RISK_RELATIONS:
                    score += 1
                if parsed_b["rel"] in self.RISK_RELATIONS:
                    score += 1

                scored_paths.append((score, f"{tuple_a} || {tuple_b}"))

        if not scored_paths:
            for tuple_str, parsed in parsed_tuples:
                tuple_entities = {parsed["subj"], parsed["obj"]}
                if parsed["rel"] not in self.RISK_RELATIONS:
                    continue
                score = 1
                if target_entity and target_entity in tuple_entities:
                    score += 3
                if focus_entities and tuple_entities & focus_entities:
                    score += 2
                scored_paths.append((score, tuple_str))

        scored_paths.sort(key=lambda item: (-item[0], len(item[1])))
        return self._dedupe_keep_order(
            [path for _, path in scored_paths]
        )[: min(6, self.max_context_tuples)]

    def _estimate_risk_score(
        self,
        risk_types: List[str],
        risk_entities: List[str],
        risk_paths: List[str],
    ) -> int:
        score = 10
        weights = {
            "credit": 20,
            "regulatory": 18,
            "contagion": 18,
            "funding": 18,
            "liquidity": 18,
            "operational": 14,
            "governance": 12,
            "reputational": 12,
            "concentration": 12,
            "market": 10,
        }

        for risk_type in risk_types:
            normalized = self._normalize_query(risk_type)
            for key, weight in weights.items():
                if key in normalized:
                    score += weight
                    break
            else:
                if normalized:
                    score += 8

        score += min(len(risk_entities), 5) * 3
        score += min(len(risk_paths), 4) * 5
        return max(0, min(100, score))

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "risk_answer": "",
            "risk_types": [],
            "risk_entities": [],
            "risk_paths": [],
            "risk_score": 0,
        }


@prompt_restrict(FinKGEntityRiskAssessmentPrompt)
@OPERATOR_REGISTRY.register()
class FinKGEntityRiskAssessment(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        k_hops: int = 2,
        max_context_tuples: int = 24,
    ):
        self.logger = get_logger()
        self.analyzer = FinKGEntityRiskAssessmentLLM(
            llm_serving=llm_serving,
            lang=lang,
            k_hops=k_hops,
            max_context_tuples=max_context_tuples,
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGEntityRiskAssessment 用于基于金融知识图谱进行实体风险预估。",
                "输入: tuple + target_entity；输出: risk_answer + risk_score。",
            )
        return (
            "FinKGEntityRiskAssessment is used to estimate entity risk with Financial KG.",
            "Input: tuple + target_entity. Output: risk_answer + risk_score.",
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key, "target_entity"]
        forbidden_keys = [self.output_key, self.output_key_score]

        if len(set(forbidden_keys)) != len(forbidden_keys):
            raise ValueError("output_key and output_key_score must be different")

        missing = [key for key in required_keys if key not in dataframe.columns]
        conflict = [key for key in forbidden_keys if key in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def process_batch(
        self,
        tuples_list: List[List[str]],
        ontology: Dict[str, Any],
        target_entities: List[Any],
    ) -> List[Dict[str, Any]]:
        raw_data = [
            {
                "tuple": tuples,
                "target_entity": target_entity,
            }
            for tuples, target_entity in zip(
                tuples_list, target_entities
            )
        ]

        return self.analyzer.llm_query(raw_data=raw_data, ontology=ontology)

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key: str = "risk_answer",
        output_key_score: str = "risk_score",
    ) -> List[str]:
        self.input_key = input_key
        self.output_key = output_key
        self.output_key_score = output_key_score

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        tuples_list = dataframe[self.input_key].tolist()
        target_entities = dataframe["target_entity"].tolist()
        ontology = load_finkg_ontology()

        outputs = self.process_batch(
            tuples_list=tuples_list,
            ontology=ontology,
            target_entities=target_entities,
        )

        dataframe[self.output_key] = [
            item.get("risk_answer", "") for item in outputs
        ]
        dataframe[self.output_key_score] = [
            item.get("risk_score", 0) for item in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Risk assessment results saved to {output_file}")

        return [self.output_key, self.output_key_score]


FinKGEntityRiskExposureAnalysis = FinKGEntityRiskAssessment
