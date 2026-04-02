import os
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class FinKGMarketauxNewsRetriever(OperatorABC):

    BASE_URL = "https://api.marketaux.com/v1"
    DEFAULT_API_TOKEN = "1Ty34059fO9mG3qmh77T50bEpBmMtNGqJQMxs0SF"

    def __init__(
        self,
        api_token: Optional[str] = None,
        request_timeout: int = 20,
        default_limit: int = 8,
        default_lookback_days: int = 7,
        default_language: str = "en",
        default_country: str = "us",
        filter_entities: bool = True,
        must_have_entities: bool = True,
        group_similar: bool = True,
    ):
        self.logger = get_logger()
        self.api_token = api_token or os.environ.get(
            "MARKETAUX_API_TOKEN",
            self.DEFAULT_API_TOKEN,
        )
        self.request_timeout = request_timeout
        self.default_limit = max(1, int(default_limit))
        self.default_lookback_days = max(1, int(default_lookback_days))
        self.default_language = default_language
        self.default_country = default_country
        self.filter_entities = filter_entities
        self.must_have_entities = must_have_entities
        self.group_similar = group_similar
        self.session = requests.Session()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGMarketauxNewsRetriever 用于从 Marketaux 外部新闻库检索目标实体的最新金融新闻。",
                "输入: target_entity，可选 symbol/country; 输出: marketaux_symbol + marketaux_news + marketaux_news_context + marketaux_avg_sentiment",
            )
        return (
            "FinKGMarketauxNewsRetriever is used to fetch recent financial news for a target entity from Marketaux.",
            "Input: target_entity with optional symbol/country; Output: marketaux_symbol + marketaux_news + marketaux_news_context + marketaux_avg_sentiment",
        )

    def run(
        self,
        storage: DataFlowStorage = None,
        input_target_key: str = "target_entity",
        input_symbol_key: Optional[str] = "symbol",
        input_country_key: Optional[str] = "country",
        output_symbol_key: str = "marketaux_symbol",
        output_name_key: str = "marketaux_entity_name",
        output_news_key: str = "marketaux_news",
        output_context_key: str = "marketaux_news_context",
        output_sentiment_key: str = "marketaux_avg_sentiment",
        output_count_key: str = "marketaux_news_count",
        lookback_days: Optional[int] = None,
        limit: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[str]:
        self.input_target_key = input_target_key
        self.input_symbol_key = input_symbol_key
        self.input_country_key = input_country_key
        self.output_symbol_key = output_symbol_key
        self.output_name_key = output_name_key
        self.output_news_key = output_news_key
        self.output_context_key = output_context_key
        self.output_sentiment_key = output_sentiment_key
        self.output_count_key = output_count_key
        self.lookback_days = lookback_days or self.default_lookback_days
        self.limit = limit or self.default_limit
        self.language = language or self.default_language

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        if not self.api_token:
            raise ValueError(
                "Missing Marketaux API token. Set MARKETAUX_API_TOKEN or pass api_token in __init__."
            )

        symbols = dataframe[self.input_symbol_key].tolist() if self.input_symbol_key and self.input_symbol_key in dataframe.columns else [None] * len(dataframe)
        countries = dataframe[self.input_country_key].tolist() if self.input_country_key and self.input_country_key in dataframe.columns else [self.default_country] * len(dataframe)

        resolved_symbols = []
        resolved_names = []
        news_rows = []
        news_contexts = []
        avg_sentiments = []
        news_counts = []

        for target_entity, symbol_value, country_value in tqdm(
            zip(dataframe[self.input_target_key].tolist(), symbols, countries),
            total=len(dataframe),
            desc="Retrieve Marketaux news",
        ):
            target_entity = self._normalize_text(target_entity)
            symbol_hint = self._normalize_text(symbol_value)
            country = self._normalize_text(country_value) or self.default_country

            resolved = self._resolve_symbol(
                target_entity=target_entity,
                symbol_hint=symbol_hint,
                country=country,
            )
            news_items = self._fetch_news(
                target_entity=target_entity,
                symbol=resolved.get("symbol", ""),
                country=country,
            )
            simplified = self._simplify_articles(
                articles=news_items,
                target_entity=target_entity,
                resolved_symbol=resolved.get("symbol", ""),
            )

            resolved_symbols.append(resolved.get("symbol", ""))
            resolved_names.append(resolved.get("name", target_entity))
            news_rows.append(simplified)
            news_contexts.append(self._build_news_context(simplified))
            avg_sentiments.append(self._average_sentiment(simplified))
            news_counts.append(len(simplified))

        dataframe[self.output_symbol_key] = resolved_symbols
        dataframe[self.output_name_key] = resolved_names
        dataframe[self.output_news_key] = news_rows
        dataframe[self.output_context_key] = news_contexts
        dataframe[self.output_sentiment_key] = avg_sentiments
        dataframe[self.output_count_key] = news_counts

        output_file = storage.write(dataframe)
        self.logger.info(f"Marketaux news retrieval results saved to {output_file}")

        return [
            self.output_symbol_key,
            self.output_name_key,
            self.output_news_key,
            self.output_context_key,
            self.output_sentiment_key,
            self.output_count_key,
        ]

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        if self.input_target_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_target_key}")

        for column in [
            self.output_symbol_key,
            self.output_name_key,
            self.output_news_key,
            self.output_context_key,
            self.output_sentiment_key,
            self.output_count_key,
        ]:
            if column in dataframe.columns:
                raise ValueError(f"Output column already exists: {column}")

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    def _request_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_params = {k: v for k, v in params.items() if v not in (None, "", [])}
        request_params["api_token"] = self.api_token
        response = self.session.get(
            f"{self.BASE_URL}{path}",
            params=request_params,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _resolve_symbol(
        self,
        target_entity: str,
        symbol_hint: str,
        country: str,
    ) -> Dict[str, str]:
        if symbol_hint:
            return {"symbol": symbol_hint, "name": target_entity}

        results = self._request_json(
            "/entity/search",
            {
                "search": target_entity,
                "countries": country,
            },
        ).get("data", [])

        if not results:
            return {"symbol": "", "name": target_entity}

        best = max(
            results,
            key=lambda item: self._entity_match_score(target_entity, item),
        )
        return {
            "symbol": best.get("symbol", ""),
            "name": best.get("name", target_entity),
        }

    def _entity_match_score(self, target_entity: str, item: Dict[str, Any]) -> float:
        name = self._normalize_text(item.get("name"))
        symbol = self._normalize_text(item.get("symbol"))
        target_norm = target_entity.lower()
        name_norm = name.lower()
        symbol_norm = symbol.lower()

        score = SequenceMatcher(None, target_norm, name_norm).ratio()
        if target_norm == name_norm:
            score += 1.0
        if target_norm in name_norm or name_norm in target_norm:
            score += 0.4
        if symbol_norm and symbol_norm == target_norm:
            score += 1.0
        return score

    def _fetch_news(
        self,
        target_entity: str,
        symbol: str,
        country: str,
    ) -> List[Dict[str, Any]]:
        lookback_days = getattr(self, "lookback_days", self.default_lookback_days)
        limit = getattr(self, "limit", self.default_limit)
        language = getattr(self, "language", self.default_language)
        published_after = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%dT%H:%M")

        params = {
            "symbols": symbol,
            "search": None if symbol else target_entity,
            "countries": country,
            "language": language,
            "published_after": published_after,
            "limit": limit,
            "filter_entities": str(self.filter_entities).lower(),
            "must_have_entities": str(self.must_have_entities).lower(),
            "group_similar": str(self.group_similar).lower(),
            "sort": "published_at",
        }
        data = self._request_json("/news/all", params)
        articles = data.get("data", []) if isinstance(data, dict) else []
        if articles:
            return articles

        if symbol:
            fallback_symbol = self._request_json(
                "/news/all",
                {
                    "symbols": symbol,
                    "language": language,
                    "published_after": published_after,
                    "limit": limit,
                    "filter_entities": str(self.filter_entities).lower(),
                    "must_have_entities": str(self.must_have_entities).lower(),
                    "group_similar": str(self.group_similar).lower(),
                    "sort": "published_at",
                },
            )
            articles = fallback_symbol.get("data", []) if isinstance(fallback_symbol, dict) else []
            if articles:
                return articles

        fallback_search = self._request_json(
            "/news/all",
            {
                "search": target_entity,
                "language": language,
                "published_after": published_after,
                "limit": limit,
                "group_similar": str(self.group_similar).lower(),
                "sort": "published_at",
            },
        )
        return fallback_search.get("data", []) if isinstance(fallback_search, dict) else []

    def _simplify_articles(
        self,
        articles: List[Dict[str, Any]],
        target_entity: str,
        resolved_symbol: str,
    ) -> List[Dict[str, Any]]:
        simplified = []

        for article in articles:
            matched_entity = self._select_article_entity(
                entities=article.get("entities", []),
                target_entity=target_entity,
                resolved_symbol=resolved_symbol,
            )

            highlights = []
            for highlight in matched_entity.get("highlights", [])[:2]:
                snippet = self._normalize_text(highlight.get("highlight"))
                if snippet:
                    highlights.append(snippet)

            simplified.append(
                {
                    "uuid": self._normalize_text(article.get("uuid")),
                    "title": self._normalize_text(article.get("title")),
                    "snippet": self._normalize_text(article.get("snippet") or article.get("description")),
                    "url": self._normalize_text(article.get("url")),
                    "source": self._normalize_text(article.get("source")),
                    "published_at": self._normalize_text(article.get("published_at")),
                    "entity_name": self._normalize_text(matched_entity.get("name")),
                    "entity_symbol": self._normalize_text(matched_entity.get("symbol")),
                    "entity_type": self._normalize_text(matched_entity.get("type")),
                    "entity_match_score": float(matched_entity.get("match_score") or 0),
                    "entity_sentiment_score": float(matched_entity.get("sentiment_score") or 0),
                    "highlights": highlights,
                }
            )

        return simplified

    def _select_article_entity(
        self,
        entities: Any,
        target_entity: str,
        resolved_symbol: str,
    ) -> Dict[str, Any]:
        if not isinstance(entities, list) or not entities:
            return {}

        def entity_score(item: Dict[str, Any]) -> float:
            name = self._normalize_text(item.get("name"))
            symbol = self._normalize_text(item.get("symbol"))
            score = 0.0
            if resolved_symbol and symbol.lower() == resolved_symbol.lower():
                score += 3.0
            score += SequenceMatcher(None, target_entity.lower(), name.lower()).ratio()
            score += float(item.get("match_score") or 0) / 100.0
            return score

        return max(entities, key=entity_score)

    def _build_news_context(self, articles: List[Dict[str, Any]]) -> str:
        if not articles:
            return ""

        lines = []
        for idx, article in enumerate(articles, start=1):
            header = (
                f"[{idx}] {article.get('published_at', 'NA')} | "
                f"{article.get('source', 'NA')} | {article.get('title', 'NA')}"
            )
            details = (
                f"entity={article.get('entity_name', 'NA')} "
                f"({article.get('entity_symbol', 'NA')}), "
                f"sentiment={article.get('entity_sentiment_score', 0):.3f}, "
                f"match={article.get('entity_match_score', 0):.3f}"
            )
            snippet = article.get("snippet", "")
            line = f"{header}\n{details}"
            if snippet:
                line += f"\nsnippet: {snippet}"
            if article.get("highlights"):
                line += f"\nhighlights: {' | '.join(article['highlights'])}"
            lines.append(line)
        return "\n\n".join(lines)

    def _average_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        if not articles:
            return 0.0
        scores = [
            float(article.get("entity_sentiment_score") or 0)
            for article in articles
        ]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)
