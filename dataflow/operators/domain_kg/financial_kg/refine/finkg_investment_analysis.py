import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.finkg import FinKGInvestmentAnalysisPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


class FinKGInvestmentAnalysisLLM:

    _REL_PATTERN = re.compile(
        r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*?)\s*<time>\s*(.*)"
    )

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC],
        lang: str = "en",
        k_hops: int = 2,
        max_context_tuples: int = 20,
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.k_hops = max(1, int(k_hops))
        self.max_context_tuples = max(1, int(max_context_tuples))
        self.logger = get_logger()
        self.prompt_template = FinKGInvestmentAnalysisPrompt(lang=lang)

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
            raise ValueError("llm_serving is required for investment analysis")

        results = []
        for data in tqdm(raw_data, desc="Analyze investment signals"):
            target_entity = self._normalize_text(data.get("target_entity"))
            market_news_context = self._normalize_text(data.get("market_news_context"))
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
                market_news_context=market_news_context,
            )
            system_prompt = self.prompt_template.build_system_prompt(ontology)

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )
            parsed = self._parse_response(responses[0] if responses else "")
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

        if not selected:
            selected = tuples[: self.max_context_tuples]

        return self._dedupe_keep_order(selected)[: self.max_context_tuples]

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
            self.logger.warning(f"Failed to parse investment analysis response: {exc}")
            return result

        result["investment_answer"] = self._safe_string(data.get("analysis_summary"))
        return result

    def _safe_string(self, value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "investment_answer": "",
        }


@prompt_restrict(FinKGInvestmentAnalysisPrompt)
@OPERATOR_REGISTRY.register()
class FinKGInvestmentAnalysis(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        k_hops: int = 2,
        max_context_tuples: int = 20,
    ):
        self.logger = get_logger()
        self.analyzer = FinKGInvestmentAnalysisLLM(
            llm_serving=llm_serving,
            lang=lang,
            k_hops=k_hops,
            max_context_tuples=max_context_tuples,
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGInvestmentAnalysis 用于基于金融知识图谱进行投资分析。",
                "输入: tuple + target_entity + marketaux_news_context；输出: investment_answer。",
            )
        return (
            "FinKGInvestmentAnalysis is used to perform investment analysis with Financial KG.",
            "Input: tuple + target_entity + marketaux_news_context. Output: investment_answer.",
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key, "target_entity", "marketaux_news_context"]
        forbidden_keys = [self.output_key]

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
        market_news_contexts: List[Any],
    ) -> List[Dict[str, Any]]:
        raw_data = [
            {
                "tuple": tuples,
                "target_entity": target_entity,
                "market_news_context": market_news_context,
            }
            for tuples, target_entity, market_news_context in zip(
                tuples_list, target_entities, market_news_contexts
            )
        ]

        return self.analyzer.llm_query(raw_data=raw_data, ontology=ontology)

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key: str = "investment_answer",
    ) -> List[str]:
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        tuples_list = dataframe[self.input_key].tolist()
        target_entities = dataframe["target_entity"].tolist()
        market_news_contexts = dataframe["marketaux_news_context"].tolist()
        ontology = load_finkg_ontology()

        outputs = self.process_batch(
            tuples_list=tuples_list,
            ontology=ontology,
            target_entities=target_entities,
            market_news_contexts=market_news_contexts,
        )

        dataframe[self.output_key] = [
            item.get("investment_answer", "") for item in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Investment analysis results saved to {output_file}")

        return [self.output_key]
