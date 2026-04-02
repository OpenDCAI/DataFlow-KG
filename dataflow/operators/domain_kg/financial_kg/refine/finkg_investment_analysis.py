import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.finkg import FinKGInvestmentAnalysisPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage

from .finkg_marketaux_news_retriever import FinKGMarketauxNewsRetriever


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
            parsed["investment_key_paths"] = self._sanitize_paths(
                paths=parsed.get("investment_key_paths", []),
                evidence_tuples=relevant_tuples,
            )
            parsed["investment_evidence_tuple"] = relevant_tuples
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

        result["investment_analysis"] = self._safe_string(data.get("analysis_summary"))
        result["bullish_signals"] = self._safe_string_list(data.get("bullish_signals"))
        result["bearish_signals"] = self._safe_string_list(data.get("bearish_signals"))
        result["watch_items"] = self._safe_string_list(data.get("watch_items"))
        result["investment_key_paths"] = self._safe_string_list(data.get("key_paths"))
        result["investment_confidence"] = self._safe_confidence(data.get("confidence"))
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

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "investment_analysis": "",
            "bullish_signals": [],
            "bearish_signals": [],
            "watch_items": [],
            "investment_key_paths": [],
            "investment_confidence": "low",
            "investment_evidence_tuple": [],
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
        self.lang = lang
        self.llm_serving = llm_serving
        self.marketaux_retriever = FinKGMarketauxNewsRetriever()
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
                "输入: target_entity + tuple，可选 symbol/country/marketaux_news_context + ontology; 输出: investment_analysis + bullish_signals + bearish_signals + watch_items",
            )
        return (
            "FinKGInvestmentAnalysis is used to perform investment analysis with Financial KG.",
            "Input: target_entity + tuple with optional symbol/country/marketaux_news_context + ontology; Output: investment_analysis + bullish_signals + bearish_signals + watch_items",
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key_tuple not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key_tuple}")
        if self.input_target_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_target_key}")

        for column in [
            self.output_summary_key,
            self.output_bullish_key,
            self.output_bearish_key,
            self.output_watch_key,
            self.output_path_key,
            self.output_confidence_key,
            self.output_evidence_key,
        ]:
            if column in dataframe.columns:
                raise ValueError(f"Output column already exists: {column}")

    def process_batch(
        self,
        tuples_list: List[List[str]],
        ontology: Dict[str, Any],
        target_entities: List[Any],
        market_news_contexts: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        if market_news_contexts is None:
            market_news_contexts = [""] * len(tuples_list)

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

    def _build_marketaux_news_contexts(
        self,
        target_entities: List[Any],
        symbols: Optional[List[Any]] = None,
        countries: Optional[List[Any]] = None,
    ) -> List[str]:
        if not self.marketaux_retriever.api_token:
            return [""] * len(target_entities)

        if symbols is None:
            symbols = [""] * len(target_entities)
        if countries is None:
            countries = [""] * len(target_entities)

        contexts = []
        for target_entity, symbol, country in zip(target_entities, symbols, countries):
            target_name = self.marketaux_retriever._normalize_text(target_entity)
            symbol_text = self.marketaux_retriever._normalize_text(symbol)
            country_text = self.marketaux_retriever._normalize_text(country) or self.marketaux_retriever.default_country

            if not target_name:
                contexts.append("")
                continue

            resolved = self.marketaux_retriever._resolve_symbol(
                target_entity=target_name,
                symbol_hint=symbol_text,
                country=country_text,
            )
            articles = self.marketaux_retriever._fetch_news(
                target_entity=target_name,
                symbol=resolved.get("symbol", ""),
                country=country_text,
            )
            simplified = self.marketaux_retriever._simplify_articles(
                articles=articles,
                target_entity=target_name,
                resolved_symbol=resolved.get("symbol", ""),
            )
            contexts.append(self.marketaux_retriever._build_news_context(simplified))

        return contexts

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists: Optional[Dict[str, Any]] = None,
        input_key_tuple: str = "tuple",
        input_target_key: str = "target_entity",
        input_symbol_key: Optional[str] = "symbol",
        input_country_key: Optional[str] = "country",
        input_news_key: Optional[str] = "marketaux_news_context",
        auto_fetch_marketaux_news: bool = True,
        input_key_meta: str = "finkg_ontology",
        output_summary_key: str = "investment_analysis",
        output_bullish_key: str = "bullish_signals",
        output_bearish_key: str = "bearish_signals",
        output_watch_key: str = "watch_items",
        output_path_key: str = "investment_key_paths",
        output_confidence_key: str = "investment_confidence",
        output_evidence_key: str = "investment_evidence_tuple",
    ) -> List[str]:
        self.input_key_tuple = input_key_tuple
        self.input_target_key = input_target_key
        self.output_summary_key = output_summary_key
        self.output_bullish_key = output_bullish_key
        self.output_bearish_key = output_bearish_key
        self.output_watch_key = output_watch_key
        self.output_path_key = output_path_key
        self.output_confidence_key = output_confidence_key
        self.output_evidence_key = output_evidence_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        tuples_list = dataframe[self.input_key_tuple].tolist()
        target_entities = dataframe[self.input_target_key].tolist()

        market_news_contexts = None
        if input_news_key and input_news_key in dataframe.columns:
            market_news_contexts = dataframe[input_news_key].tolist()
        elif auto_fetch_marketaux_news:
            symbols = None
            countries = None
            if input_symbol_key and input_symbol_key in dataframe.columns:
                symbols = dataframe[input_symbol_key].tolist()
            if input_country_key and input_country_key in dataframe.columns:
                countries = dataframe[input_country_key].tolist()
            market_news_contexts = self._build_marketaux_news_contexts(
                target_entities=target_entities,
                symbols=symbols,
                countries=countries,
            )

        if ontology_lists is None:
            storage_meta = FileStorage(first_entry_file_name="", cache_type="json")
            ontology_df = storage_meta.read(
                file_path=f"./.cache/api/{input_key_meta}.json",
                output_type="dataframe",
            )
            row = ontology_df.iloc[0]
            ontology = {
                "entity_type": row["entity_type"],
                "relation_type": row["relation_type"],
                "attribute_type": row.get("attribute_type", {}),
            }
        else:
            ontology = ontology_lists

        outputs = self.process_batch(
            tuples_list=tuples_list,
            ontology=ontology,
            target_entities=target_entities,
            market_news_contexts=market_news_contexts,
        )

        dataframe[self.output_summary_key] = [
            item.get("investment_analysis", "") for item in outputs
        ]
        dataframe[self.output_bullish_key] = [
            item.get("bullish_signals", []) for item in outputs
        ]
        dataframe[self.output_bearish_key] = [
            item.get("bearish_signals", []) for item in outputs
        ]
        dataframe[self.output_watch_key] = [
            item.get("watch_items", []) for item in outputs
        ]
        dataframe[self.output_path_key] = [
            item.get("investment_key_paths", []) for item in outputs
        ]
        dataframe[self.output_confidence_key] = [
            item.get("investment_confidence", "low") for item in outputs
        ]
        dataframe[self.output_evidence_key] = [
            item.get("investment_evidence_tuple", []) for item in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Investment analysis results saved to {output_file}")

        return [
            self.output_summary_key,
            self.output_bullish_key,
            self.output_bearish_key,
            self.output_watch_key,
            self.output_path_key,
            self.output_confidence_key,
            self.output_evidence_key,
        ]
